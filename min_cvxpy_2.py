import os
import sys
import time
import numpy as np
import cvxpy as cp
import qutip as qt
import scipy.linalg as la
import scipy.sparse as sp

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

k=0.5

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
BASE_PATH = "molecular_model/"  
OUTPUT_PATH = f"molecular_model/results_sdp/{k}"

OS_KS_PATH = os.path.join(OUTPUT_PATH, "Ks")

os.makedirs(OUTPUT_PATH, exist_ok=True)
for path in [OS_KS_PATH]:
    os.makedirs(path, exist_ok=True)

# Parámetros físicos del sistema
N_c = 5 
S = 0.5 
N_Q = int(2*S+1)


I_c = qt.qeye(N_c)
I_Q = qt.qeye(N_Q)
a = qt.destroy(N_c)
a_q = qt.tensor(a, I_c)
a_d_q = a_q.dag()
a_p = qt.tensor(I_c, a)
a_d_p = a_p.dag()

Q = 1/np.sqrt(2)*(a_q+a_d_q)
P = 1/np.sqrt(2)*(a_p+a_d_p)
Pi_Q = -1j/np.sqrt(2)*(a_q-a_d_q)
Pi_P = -1j/np.sqrt(2)*(a_p-a_d_p)


# Parámetro de regularización L1 para favorecer matrices de Kossakowski ralas
LAMBDA_REG = 1e-5 

# ==============================================================================
# 2. FUNCIONES AUXILIARES Y GENERACIÓN DE BASES
# ==============================================================================
H = np.load(BASE_PATH + f"hamiltonians/hamiltonian_k_{k}_2.npy")
rho0 = np.array((qt.qload(BASE_PATH + f"HCEs/k_{k}/HCE_1.0000")).full())
rho0 = rho0 / np.trace(rho0)

# ==============================================================================
# 2. CREACIÓN DE LAS BASES (Con la Base Global para Espines)
# ==============================================================================
def basis_creator(max_g):
    print(f"Generando base para orden máximo {max_g}...")
    basis_pi = []
    monomios_fase_total = [] 
    
    for order in range(max_g, -1, -1): 
        monomios_orden = []
        for exp_Q in range(order, -1, -1):
            exp_P = order - exp_Q
            monomios_orden.append(Q**exp_Q * P**exp_P)
            
        monomios_fase_total.extend(monomios_orden)
        
        # Base Pi (Igual que antes)
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(qt.identity(2), qt.identity(2), monomio))
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(qt.identity(2), qt.identity(2), monomio * Pi_Q))
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(qt.identity(2), qt.identity(2), monomio * Pi_P))

    spin_ops = [
        (qt.sigmax(), qt.identity(2)), (qt.sigmay(), qt.identity(2)), (qt.sigmaz(), qt.identity(2)),
        (qt.identity(2), qt.sigmax()), (qt.identity(2), qt.sigmay()), (qt.identity(2), qt.sigmaz()),
        (qt.sigmax(), qt.sigmax()), (qt.sigmax(), qt.sigmay()), (qt.sigmax(), qt.sigmaz()),
        (qt.sigmay(), qt.sigmax()), (qt.sigmay(), qt.sigmay()), (qt.sigmay(), qt.sigmaz()),
        (qt.sigmaz(), qt.sigmax()), (qt.sigmaz(), qt.sigmay()), (qt.sigmaz(), qt.sigmaz())
    ]

    # NUEVO: Aplanamos TODO en una única super-base global
    basis_spin_global = []
    for op1, op2 in spin_ops:
        for monomio in monomios_fase_total:
            basis_spin_global.append(qt.tensor(op1, op2, monomio))
            
    return basis_pi, basis_spin_global

max_orden_polinomio = int(sys.argv[1])
basis_pi, basis_spin_global = basis_creator(max_orden_polinomio)

# ==============================================================================
# 3. PRECÁLCULO DE LOS TENSORES DE KOSSAKOWSKI
# ==============================================================================
def precalcular_tensor(basis_list, rho0_np):
    N = len(basis_list)
    dim = rho0_np.shape[0]
    A_ij = np.zeros((N, N, dim, dim), dtype=np.complex128)
    
    B_mats = [B.full() for B in basis_list]
    B_dags = [B.conj().T for B in B_mats]
    
    for i in range(N):
        for j in range(N):
            jump = B_mats[i] @ rho0_np @ B_dags[j]
            decay = -0.5 * (B_dags[j] @ B_mats[i] @ rho0_np + rho0_np @ B_dags[j] @ B_mats[i])
            A_ij[i, j] = jump + decay
            
    return A_ij

print(f"\nPrecalculando tensores cruzados...")
print(f"Tamaño base Pi: {len(basis_pi)}")
print(f"Tamaño base Global de Espín: {len(basis_spin_global)} (Todas las combinaciones)")

A_ij_pi = precalcular_tensor(basis_pi, rho0)
A_ij_spin = precalcular_tensor(basis_spin_global, rho0)

import scipy.sparse as sp

# ==============================================================================
# 4. VECTORIZACIÓN 1D Y COMPRESIÓN DISPERSA (EL SALVA-RAM)
# ==============================================================================
def precalcular_vectores_dispersos(A_ij_array):
    """
    Aplana el tensor 4D en una matriz 2D dispersa que enlaza perfectamente
    con el comando cp.vec(chi, order='F') de CVXPY.
    """
    N = A_ij_array.shape[0]
    dim = A_ij_array.shape[2]
    
    # Matriz plana gigante
    A_flat = np.zeros((N * N, dim * dim), dtype=np.complex128)
    
    k = 0
    # Orden Fortran: recorremos filas (i) y luego columnas (j)
    for j in range(N):
        for i in range(N):
            # Aplanamos el operador dim x dim en un vector 1D
            A_flat[k, :] = A_ij_array[i, j].flatten(order='F')
            k += 1
            
    # FILTRADO AGRESIVO: Convertimos a 0 el ruido numérico
    A_flat[np.abs(A_flat) < 1e-12] = 0.0
    
    # Convertimos a formato disperso (solo guarda los no-ceros en RAM)
    A_sparse = sp.csc_matrix(A_flat)
    
    porcentaje = (A_sparse.nnz / (N * N * dim * dim)) * 100
    print(f"    -> Memoria salvada: {A_sparse.nnz} elementos de {N*N*dim*dim} ({porcentaje:.2f}% de densidad)")
    
    return A_sparse

print("\nComprimiendo tensores a matrices dispersas...")
A_pi_sparse = precalcular_vectores_dispersos(A_ij_pi)
A_spin_sparse = precalcular_vectores_dispersos(A_ij_spin)

# ==============================================================================
# 5. OPTIMIZACIÓN CONVEXA 1D (CON VARIABLES DE HOLGURA)
# ==============================================================================
print("\nConfigurando el problema en CVXPY (Formato 1D Vectorizado)...")
lambda_reg = 1e-4
dim = rho0.shape[0]
dim_sq = dim * dim

# 1. Variables de optimización físicas (Matrices de Kossakowski)
chi_pi = cp.Variable((len(basis_pi), len(basis_pi)), hermitian=True)
chi_spin = cp.Variable((len(basis_spin_global), len(basis_spin_global)), hermitian=True)

# 2. Variables de Holgura (Slack) 1D para el error. 
# Esto evita que CVXPY colapse intentando elevar al cuadrado expresiones complejas.
E_real = cp.Variable(dim_sq)
E_imag = cp.Variable(dim_sq)

# 3. Construcción Vectorizada del Liouvilliano Total
# Aplanamos el Hamiltoniano inicial
L_ham = -1j * (H @ rho0 - rho0 @ H)
L_ham_vec = L_ham.flatten(order='F')

# Multiplicación súper eficiente: Matriz Dispersa Transpuesta * Vector de Variables
L_pi_vec = A_pi_sparse.T @ cp.vec(chi_pi, order='F')
L_spin_vec = A_spin_sparse.T @ cp.vec(chi_spin, order='F')

L_total_vec = L_ham_vec + L_pi_vec + L_spin_vec

# 4. Restricciones
restricciones = [
    chi_pi >> 0,
    chi_spin >> 0,
    cp.real(L_total_vec) == E_real,  # Desacople perfecto
    cp.imag(L_total_vec) == E_imag
]

# 5. Función de coste (Minimizamos los vectores de holgura simples)
error_frob_sq = cp.sum_squares(E_real) + cp.sum_squares(E_imag)
traza_total = cp.trace(chi_pi) + cp.trace(chi_spin)
coste = error_frob_sq + lambda_reg * traza_total

start_time=time.time()
# 6. Resolver
problema = cp.Problem(cp.Minimize(coste), restricciones)
print("Resolviendo con SCS...")
problema.solve(solver=cp.SCS, verbose=True)

print(f"\nCoste óptimo final: {problema.value:.6e}. Tiempo empleado: {time.time()-start_time} s.")
with open(OUTPUT_PATH+"/datos.csv", "a") as f:
    f.write(f"{max_orden_polinomio},{time.time()-start_time},{problema.value:.6f}\n")

# ==============================================================================
# 6. EXTRACCIÓN DE LOS OPERADORES GLOBALES
# ==============================================================================
def extraer_coefs(chi_matrix_val, basis_list, threshold=1e-6):
    """
    Extrae los operadores K y sus coeficientes a partir de la matriz chi evaluada.
    """
    evals, evecs = la.eigh(chi_matrix_val)
    
    #K_ops = []
    coefs_list = []
    
    for i in range(len(evals)):
        #if evals[i] > threshold:
        # El coeficiente final incluye el peso (la raíz del autovalor)
        peso = np.sqrt(evals[i])
        coeficientes = peso * evecs[:, i]
        
        K = np.zeros_like(basis_list[0].full(), dtype=np.complex128)
        for j in range(len(basis_list)):
            K += coeficientes[j] * basis_list[j].full()
            
        #K_ops.append(K)
        coefs_list.append(coeficientes)
            
    # Convertimos la lista de coeficientes a un array de numpy 2D (Num_K x Tamaño_Base)
    return np.array(coefs_list)

def guardar_coeficientes(archivo_salida, coefs_pi, coefs_spin):
    """
    Guarda los coeficientes de los polinomios en un archivo .npz
    """
    np.savez(archivo_salida, coefs_pi=coefs_pi, coefs_spin=coefs_spin)
    print(f"\nCoeficientes guardados exitosamente en: {archivo_salida}.npz")

def leer_y_reconstruir_K(archivo_entrada, basis_pi, basis_spin_global):
    """
    Lee un archivo .npz y reconstruye los operadores de salto K.
    """
    # Cargamos el archivo (añadimos .npz si el usuario no lo puso)
    if not archivo_entrada.endswith('.npz'):
        archivo_entrada += '.npz'
        
    datos = np.load(archivo_entrada)
    coefs_pi = datos['coefs_pi']
    coefs_spin = datos['coefs_spin']
    
    K_ops_pi = []
    K_ops_spin = []
    
    # Reconstruimos los operadores Pi
    if coefs_pi.ndim > 1: # Verifica que no esté vacío
        for coeficientes in coefs_pi:
            K = np.zeros_like(basis_pi[0].full(), dtype=np.complex128)
            for j in range(len(basis_pi)):
                K += coeficientes[j] * basis_pi[j].full()
            K_ops_pi.append(K)
            
    # Reconstruimos los operadores Spin
    if coefs_spin.ndim > 1:
        for coeficientes in coefs_spin:
            K = np.zeros_like(basis_spin_global[0].full(), dtype=np.complex128)
            for j in range(len(basis_spin_global)):
                K += coeficientes[j] * basis_spin_global[j].full()
            K_ops_spin.append(K)
            
    print(f"\nArchivo cargado: {archivo_entrada}")
    print(f"-> Operadores Pi reconstruidos: {len(K_ops_pi)}")
    print(f"-> Operadores Spin reconstruidos: {len(K_ops_spin)}")
    
    return K_ops_pi, K_ops_spin, coefs_pi, coefs_spin

print("\nExtrayendo operadores de salto resultantes...")

K_ops_pi = extraer_coefs(chi_pi.value, basis_pi)
K_ops_spin = extraer_coefs(chi_spin.value, basis_spin_global)
guardar_coeficientes(OUTPUT_PATH+f"/Ks/{max_orden_polinomio}", K_ops_pi, K_ops_spin)

print(f"-> Canales Pi (f, g, h mezclados): {len(K_ops_pi)}")
print(f"-> Canales Globales de Espín (polinomios y espines entrelazados): {len(K_ops_spin)}")
print(f"\nNUM_K TOTAL GENERADO: {len(K_ops_pi) + len(K_ops_spin)}")