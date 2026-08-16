import numpy as np
from sympy import *
import math
from scipy import special
from qutip import Qobj, qsave
from os import makedirs
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ==========================================
# 0. FUNCIONES AUXILIARES
# ==========================================
def phi_n(n, x):
    return 1/math.sqrt(2**n*math.factorial(n))*math.pi**(-0.25)*np.exp(-np.array(x)**2/2)*special.eval_hermite(n, x)

# LA NUEVA FUNCIÓN MAESTRA QUE PROCESA UNA BETA ENTERA
def procesar_beta(beta, evals, evecs, phi_evals, Delta, Nc2, k_val, dim, N_Q, q_grid, p_grid):
    NQ2 = N_Q**2
    
    # 1. Matriz exponencial y función de partición
    exp_evals = np.exp(-beta * evals) 
    expH_grid = np.einsum('xyik,xyk,xyjk->ijxy', evecs, exp_evals, evecs.conj(), optimize=True)
    
    tr_grid = np.trace(expH_grid, axis1=0, axis2=1).real
    Z = np.sum(tr_grid) * Delta**2

    # 2. Gráfico (solo si beta = 1)
    # Matplotlib no es muy amigo de los hilos, usamos el backend 'Agg' por seguridad
    if np.isclose(beta, 1.0, atol=1e-4):
        plt.switch_backend('Agg')
        fig = plt.figure(figsize=(8, 6))
        mapa = plt.pcolormesh(q_grid, p_grid, tr_grid, cmap='viridis', shading='auto')
        plt.colorbar(mapa)
        plt.title(rf'Tr$\left(e^{{-\beta H(q,p)}}\right)$, $k={k_val}$')
        plt.xlabel(r'$q$')
        plt.ylabel(r'$p$')
        plt.tight_layout()
        plt.savefig(f"molecular_model/trace_k={k_val}.png", dpi=300)
        plt.close(fig)

    # 3. Matriz densidad local
    rho_xi = expH_grid / Z
    del expH_grid, tr_grid, evecs # Liberar memoria crucial

    # 4. Cálculo secuencial rápido de los 16 bloques HCE
    rho_HCE = np.zeros((dim, dim), dtype='complex128')
    
    for m in range(NQ2):
        for mp in range(NQ2):
            rho_val = np.sqrt(rho_xi[m, mp].astype('complex128'))
            B = np.einsum('ai,ij,bj->ab', phi_evals, rho_val, phi_evals, optimize=True) * Delta**2
            B2 = np.einsum('ab,cd->abcd', B, B)
            rho_HCE[Nc2*m:Nc2*(m+1), Nc2*mp:Nc2*(mp+1)] = B2.reshape(Nc2, Nc2)
            
    # 5. Guardado QuTiP
    N_c = int(np.sqrt(Nc2))
    rho_qobj = Qobj(rho_HCE)
    rho_qobj.dims = [[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]]

    qsave(rho_qobj, f"molecular_model/HCEs/k_{k_val}/HCE_{beta:.4f}")
    
    return f"Beta {beta:.4f} completada"

# Crear directorios
makedirs("molecular_model/HCEs", exist_ok=True)

# ==========================================
# 1. PARÁMETROS Y EVALUACIÓN SIMBÓLICA (Abreviado aquí, igual que tu código)
# ==========================================
q_max = 10 
N_points = 3000 
Delta = 2 * q_max / (N_points - 1)
k_val = 1.0   
betas = np.logspace(-2, 2, 50)
N_c = 5
N_Q = 2
NQ2 = N_Q**2
Nc2 = N_c**2
dim = Nc2 * NQ2

makedirs(f"molecular_model/HCEs/k_{k_val}", exist_ok=True)

qp_list = np.linspace(-q_max, q_max, N_points)
q_grid, p_grid = np.meshgrid(qp_list, qp_list, indexing='ij')

q, p, A, B_sym, C, D, k_sym, E, T = symbols("q p A B C D k E T")
Afun = Piecewise((A * (1 - exp(-B_sym*q)), q > 0), (A * (exp(B_sym*q) - 1), True)) 
Cfun = C * exp(-D*q**2)
H_sym = Matrix([[0.5*k_sym*(p**2+q**2)+2*E, T, T, 0], [T, 0.5*k_sym*(p**2+q**2), 0, T], [T, 0, 0.5*k_sym*(p**2+q**2), T], [0, T, T, 0.5*k_sym*(p**2+q**2)-2*E]])

parametros = {A: 0.01, B_sym: 1.6, C: 0.005, D: 1.0, k_sym: k_val}
H_sub = H_sym.subs({E: Afun, T: Cfun}).subs(parametros)
H_tuple = tuple(tuple(H_sub[i, j] for j in range(NQ2)) for i in range(NQ2))
H_func = lambdify((q, p), H_tuple, modules='numpy')

res = H_func(q_grid, p_grid)
H_grid = np.empty((NQ2, NQ2, N_points, N_points))
for i in range(NQ2):
    for j in range(NQ2):
        H_grid[i, j] = res[i][j]
del res

phi_evals = np.array([[phi_n(n, x) for x in qp_list] for n in range(N_c)])
evals, evecs = np.linalg.eigh(np.moveaxis(H_grid, [0, 1], [-2, -1])) 
del H_grid

# ==========================================
# LA MAGIA: PARALELIZAR LAS BETAS DE GOLPE
# ==========================================
print("Lanzando cálculos paralelos sobre todas las temperaturas...")

# Usa prefer="processes" o omítelo si tienes RAM de sobra. 
# Usa prefer="threads" si andas corto de RAM y quieres forzar compartir memoria, 
# aunque procesos puros escala un pelín mejor si la RAM lo permite.
resultados = Parallel(n_jobs=4, require='sharedmem', verbose=10)(
    delayed(procesar_beta)(beta, evals, evecs, phi_evals, Delta, Nc2, k_val, dim, N_Q, q_grid, p_grid)
    for beta in betas
)

print("Proceso finalizado.")