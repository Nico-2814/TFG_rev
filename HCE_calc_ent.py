import numpy as np
from sympy import *
import math
from scipy import special
from qutip import Qobj, qsave
from os import makedirs
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib import rcParams
rcParams.update({'font.size': 16})

# ==========================================
# 0. FUNCIONES AUXILIARES
# ==========================================
def phi_n(n, x):
    return 1/math.sqrt(2**n*math.factorial(n))*math.pi**(-0.25)*np.exp(-np.array(x)**2/2)*special.eval_hermite(n, x)

def calcular_bloque_einsum(m, mp, rho_val, phi_p, D2, N2):
    B = np.einsum('ni,ij,mj->nm', phi_p, np.sqrt(rho_val.astype('complex128')), phi_p, optimize=True) * D2
    B2=np.einsum('ab,cd->abcd', B, B, optimize=True)
    return m, mp, B2.reshape(N2, N2)

# Crear directorios
makedirs("molecular_model", exist_ok=True)
makedirs("molecular_model/HCEs_no_diag", exist_ok=True)

# ==========================================
# 1. PARÁMETROS FÍSICOS Y DE LA CUADRÍCULA
# ==========================================
q_max = 10 
N_points = 3000 
Delta = 2 * q_max / (N_points - 1)

k_val = 1   # Tu parámetro k
#betas = np.logspace(-0.2, 0.2, 50)    # Beta para la gráfica y matriz densidad
betas=[1]
N_Q = 2
NQ2 = N_Q**2  # NQ2 = 4 (Dimensión de la matriz)

makedirs(f"molecular_model/HCEs/k_{k_val}", exist_ok=True)

qp_list = np.linspace(-q_max, q_max, N_points)
q_grid, p_grid = np.meshgrid(qp_list, qp_list, indexing='ij')

# ==========================================
# 2. DEFINICIÓN SIMBÓLICA (Nuevo Modelo)
# ==========================================
q, p, A, B, C, D, k_sym, E, T = symbols("q p A B C D k E T")

Afun = Piecewise(
    (A * (1 - exp(-B*q)), q > 0),
    (A * (exp(B*q) - 1), True)
) 
Cfun = C * exp(-D*q**2)

H_sym = Matrix([
    [0.5*k_sym*(p**2+q**2)+2*E, T, T, 0], 
    [T, 0.5*k_sym*(p**2+q**2), 0, T], 
    [T, 0, 0.5*k_sym*(p**2+q**2), T], 
    [0, T, T, 0.5*k_sym*(p**2+q**2)-2*E]
])

parametros = {
    A: 0.01,
    B: 1.6,
    C: 0.005,
    D: 1.0,
    k_sym: k_val
}

# Sustituimos todo y preparamos lambdify
H_sub = H_sym.subs({E: Afun, T: Cfun}).subs(parametros)
H_tuple = tuple(tuple(H_sub[i, j] for j in range(NQ2)) for i in range(NQ2))
H_func = lambdify((q, p), H_tuple, modules='numpy')

# ==========================================
# 3. EVALUACIÓN Y EXPONENCIAL NUMÉRICA
# ==========================================
print("Evaluando matriz H en la cuadrícula...")
res = H_func(q_grid, p_grid)
H_grid = np.empty((NQ2, NQ2, N_points, N_points))
for i in range(NQ2):
    for j in range(NQ2):
        H_grid[i, j] = res[i][j]
del res

with open("molecular_model/entropies_2.csv", "w") as f:
            

    print("Diagonalizando H_grid...")
    # Diagonalización vectorizada a lo largo de la cuadrícula
    evals, evecs = np.linalg.eigh(np.moveaxis(H_grid, [0, 1], [-2, -1])) 

    # ==========================================
    # 4. BUCLE PARA CADA TEMPERATURA (BETA)
    # ==========================================
    for beta in betas:
        print(f"Calculando Matriz Exponencial para beta={beta:.4f}...")
        exp_evals = np.exp(-beta * evals) 
        expH_grid = np.moveaxis((evecs * exp_evals[..., None, :]) @ np.swapaxes(evecs.conj(), -1, -2), [-2, -1], [0, 1])
        del exp_evals

        tr_grid = np.trace(expH_grid, axis1=0, axis2=1).real # .real por seguridad numérica
        Z = np.sum(tr_grid) * Delta**2

        
        # Matriz densidad local
        rho_xi = expH_grid / Z
        del expH_grid, tr_grid

        # Creamos la matriz identidad de tamaño NQ2xNQ2
        # y le añadimos dos ejes extra al final para que su shape sea (NQ2, NQ2, 1, 1)
        

        for N_c in range(3,50):
            Nc2 = N_c**2
            dim = Nc2 * NQ2

            # Se calcula phi_evals ANTES del bucle beta, ahora que phi_n ya existe
            phi_evals = np.array([[phi_n(n, x) for x in qp_list] for n in range(N_c)])
            # Inicializamos la Matriz global HCE
            rho_HCE = np.zeros((dim, dim), dtype='complex128')

            # Paralelización
            resultados = Parallel(n_jobs=4, prefer="threads")(
                delayed(calcular_bloque_einsum)(m, mp, rho_xi[m, mp], phi_evals, Delta**2, Nc2)
                for m in range(NQ2) for mp in range(NQ2)
            )
            
            for m, mp, B_2d in resultados:
                # ATENCIÓN: B_2d * B_2d es elemento a elemento. Usa B_2d @ B_2d si querías producto matricial.
                rho_HCE[Nc2*m:Nc2*(m+1), Nc2*mp:Nc2*(mp+1)] = B_2d

            trace_HCE=rho_HCE.trace().real
            print(f"{trace_HCE}\n")
            Z_id = np.log(Z*trace_HCE) * np.eye(NQ2)[:, :, np.newaxis, np.newaxis]
        
            # Sumamos H_grid y Z_id (el broadcasting funciona automáticamente aquí)
            termino_parentesis = H_grid + Z_id
        
            # ¡ATENCIÓN AL PRODUCTO!
            # Recordando nuestra charla anterior sobre el teorema: los índices discretos 
            # (las dos primeras dimensiones) deben multiplicarse de forma MATRICIAL. 
            # Si usas el operador '*', NumPy hará una multiplicación elemento a elemento.
            # Para hacer un producto matricial en los dos primeros índices a lo largo de toda la malla:
            kernel = -1 * np.einsum('ij...,jk...->ik...', rho_xi/trace_HCE, termino_parentesis)

            # Paralelización
            resultados = Parallel(n_jobs=4, prefer="threads")(
                delayed(calcular_bloque_einsum)(m, mp, kernel[m, mp], phi_evals, Delta**2, Nc2)
                for m in range(NQ2) for mp in range(NQ2)
            )

            for m, mp, B_2d in resultados:
                # ATENCIÓN: B_2d * B_2d es elemento a elemento. Usa B_2d @ B_2d si querías producto matricial.
                rho_HCE[Nc2*m:Nc2*(m+1), Nc2*mp:Nc2*(mp+1)] = B_2d 
                
            rho_qobj = Qobj(rho_HCE)
            rho_qobj.dims = [[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]]

            f.write(f"{N_c},{rho_qobj.tr().real}\n")

            #qsave(rho_qobj, f"molecular_model/HCEs_no_diag/k_{k_val}/HCE_{beta:.4f}")
        
    print("Proceso finalizado correctamente.")