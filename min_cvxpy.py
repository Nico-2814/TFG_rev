import os
import sys
import time
import numpy as np
import cvxpy as cp
import qutip as qt

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
BASE_PATH = "molecular_model/"  
OUTPUT_PATH = "/kaggle/working/results_sdp/"

OS_KS_PATH = os.path.join(OUTPUT_PATH, "Ks")
OS_EVALS_PATH = os.path.join(OUTPUT_PATH, "evals")
OS_RHOS_PATH = os.path.join(OUTPUT_PATH, "rhos")

for path in [OS_KS_PATH, OS_EVALS_PATH, OS_RHOS_PATH]:
    os.makedirs(path, exist_ok=True)

# Parámetros físicos del sistema
N_c = 5 
S = 0.5 
N_Q = int(2*S+1)
k=0.5

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
H_np = np.load(BASE_PATH + f"hamiltonians/hamiltonian_k_{k}_2.npy")
rho_0np = np.array((qt.qload(BASE_PATH + f"HCEs/k_{k}/HCE_1.0000")).full())
rho_0np = rho_0np / np.trace(rho_0np)

def basis_creator(max_g):
    print(f"\nPrecalculando tensores base para orden {max_g}...")
    basis_pi = []
    monomios_fase_total = [] 
    
    for order in range(max_g, -1, -1): 
        monomios_orden = []
        for exp_Q in range(order, -1, -1):
            exp_P = order - exp_Q
            monomios_orden.append(Q**exp_Q * P**exp_P)
            
        monomios_fase_total.extend(monomios_orden)
        
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(I_Q, I_Q, monomio))
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(I_Q, I_Q, monomio * Pi_Q))
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(I_Q, I_Q, monomio * Pi_P))

    spin_ops = [
        (qt.sigmax(), I_Q), (qt.sigmay(), I_Q), (qt.sigmaz(), I_Q),
        (I_Q, qt.sigmax()), (I_Q, qt.sigmay()), (I_Q, qt.sigmaz()),
        (qt.sigmax(), qt.sigmax()), (qt.sigmax(), qt.sigmay()), (qt.sigmax(), qt.sigmaz()),
        (qt.sigmay(), qt.sigmax()), (qt.sigmay(), qt.sigmay()), (qt.sigmay(), qt.sigmaz()),
        (qt.sigmaz(), qt.sigmax()), (qt.sigmaz(), qt.sigmay()), (qt.sigmaz(), qt.sigmaz())
    ]

    basis_spin = []
    for op1, op2 in spin_ops:
        op_basis = []
        for monomio in monomios_fase_total:
            op_basis.append(qt.tensor(op1, op2, monomio))
        basis_spin.extend(op_basis)
            
    return basis_pi + basis_spin

# ==============================================================================
# 3. SOLVER DE PROGRAMACIÓN SEMIDEFINIDA (SDP)
# ==============================================================================
import numpy as np
import cvxpy as cp
import time

def resolver_cvxpy_diagonal(H_np, rho0_np, basis_matrices, lambda_reg=1e-5):
    """
    Encuentra los coeficientes gamma_i y extrae los operadores K_i
    minimizando ||L(rho0)||_F asumiendo una base de operadores DIAGONAL.
    """
    N = H_np.shape[0]
    M = len(basis_matrices)
    print("\n==================================================")
    print(f"      EJECUTANDO SOLVER DIAGONAL (CVXPY - M={M})      ")
    print("==================================================")

    start_time = time.time()

    # 1. Término Hamiltoniano: A0 = -i [H, rho0]
    L_rho0 = -1j * (H_np @ rho0_np - rho0_np @ H_np)

    # 2. Precálculo de tensores de respuesta A_i (Bucle único)
    print("Precalculando respuesta de la base (A_i)...")
    A_i = np.zeros((M, N, N), dtype=np.complex128)
    gamma = cp.Variable(M, nonneg=True)
    for i in range(M):
        Bi = basis_matrices[i].full()
        Bi_dag = Bi.conj().T
        
        jump = Bi @ rho0_np @ Bi_dag
        decay = -0.5 * (Bi_dag @ Bi @ rho0_np + rho0_np @ Bi_dag @ Bi)
        L_rho0 = L_rho0 + gamma[i] * (jump + decay)


    # 5. Función de coste convexa
    coste_norma = cp.norm(L_rho0, "fro")
    # La traza de una matriz diagonal es simplemente la suma de sus elementos
    regularizacion = lambda_reg * cp.sum(gamma)
    objetivo = cp.Minimize(coste_norma + regularizacion)

    # 6. Resolución del problema de optimización
    problema = cp.Problem(objetivo)
    print("Resolviendo problema de optimización convexa en CPU...")
    
    # ECOS y OSQP son solvers excelentes y ultra rápidos para esto
    try:
        problema.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        try:
            problema.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            problema.solve(solver=cp.SCS, verbose=False)

    print(f" Estado del Solver: {problema.status}")
    print(f" Error alcanzado ||L(rho0)||_F: {coste_norma.value:.8e}")
    print(f" Tiempo de optimización: {time.time() - start_time:.2f} s")

    # 7. Construcción directa de Operadores K_i (sin diagonalización)
    gamma_opt = gamma.value
    jump_ops = []
    
    # Control de seguridad por si el solver fallara
    if gamma_opt is None:
        print(" Error: El solver no convergió a una solución óptima.")
        return [], None, None

    
    
    # Devolvemos diag(gamma_opt) si el resto de tu código espera una matriz de coeficientes
    return gamma_opt, coste_norma.value

# ==============================================================================
# 4. EVALUACIÓN Y GUARDADO DE RESULTADOS
# ==============================================================================
def evaluar_y_guardar(H_np, rho0_np, basis, gamma, order=5):
    """Construye el Liouvilliano completo, calcula la respuesta final y guarda a disco."""
    N = H_np.shape[0]
    M = len(basis_matrices)
    I_N = np.eye(N, dtype=np.complex128)

    print("\nConstruyendo Liouvilliano explícito para validación...")
    
    # Construcción directa en NumPy
    H_super = -1j * (np.kron(H_np, I_N) - np.kron(I_N, H_np.T))
    jump_ops=[]
    for i in range(M):
        jump_ops.append(np.sqrt(gamma[i])*basis[i].full())
    
    L_diss = np.zeros((N*N, N*N), dtype=np.complex128)
    for K in jump_ops:
        K_dag = K.conj().T
        K_tail = K_dag @ K
        term_jump = np.kron(K, K.conj())
        term_tail = -0.5 * np.kron(K_tail, I_N) - 0.5 * np.kron(I_N, K_tail.T)
        L_diss += term_jump + term_tail

    L = H_super + L_diss

    # Cálculo del estado estacionario efectivo
    L_mod = np.copy(L)
    identity_vec = np.eye(N, dtype=np.complex128).flatten()
    L_mod[0, :] = identity_vec
    rhs = np.zeros(N * N, dtype=np.complex128)
    rhs[0] = 1.0

    rho_steady_vec = np.linalg.solve(L_mod, rhs)
    rho_steady = rho_steady_vec.reshape((N, N))

    # Métricas de error
    rho0qt=qt.Qobj(rho_0np, dims=[[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]])
    distancia_norma = np.linalg.norm(rho0_np - rho_steady)
    print(f" Distancia final ||rho_steady - rho0||_F: {distancia_norma:.8e}")
    rho_qt = qt.Qobj(np.array(rho_steady), dims=[[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]])
    print(f"-> Fidelidad: {qt.fidelity(rho_qt, rho0qt)}")
    print(f"-> Distancia de traza: {qt.tracedist(rho_qt, rho0qt)}")

    # Autovalores del Liouvilliano (en CPU)
    print("Calculando autovalores del Liouvilliano...")
    evals_L = np.linalg.eigvals(L)

    # Guardar archivos
    np.save(os.path.join(OS_EVALS_PATH, f"liouvillian_evals_sdp_g{order}.npy"), evals_L)
    np.save(os.path.join(OS_RHOS_PATH, f"rho_steady_sdp_g{order}.npy"), rho_steady)
    np.save(os.path.join(OS_KS_PATH, f"gamma_g{order}.npy"), gamma)

    # Guardar operadores K_k
    for idx, K in enumerate(jump_ops):
        np.save(os.path.join(OS_KS_PATH, f"K_sdp_g{order}_channel_{idx}.npy"), K)

    print(f" ¡Todos los resultados guardados en {OUTPUT_PATH}!")

# ==============================================================================
# 5. PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("\n==================================================")
    print("    INICIANDO SIMULACIÓN DE LIOUVILLE VÍA SDP     ")
    print("==================================================")

    # 1. Cargar Hamiltoniano y estado objetivo


    # 2. Generar base de monomios
    ORDEN_MAXIMO = 8
    basis_matrices = basis_creator(ORDEN_MAXIMO)

    # 3. Resolver vía SDP
    gamma_opt, coste_sdp = resolver_cvxpy_diagonal(
        H_np, rho_0np, basis_matrices, lambda_reg=LAMBDA_REG
    )

    # 4. Validar y guardar resultados
    evaluar_y_guardar(H_np, rho_0np, basis_matrices, gamma_opt, order=ORDEN_MAXIMO)

    print("\n==================================================")
    print("           SIMULACIÓN SDP FINALIZADA              ")
    print("==================================================")