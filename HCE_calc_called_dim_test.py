import numpy as np
from sympy import *
import math
from scipy import special
from qutip import Qobj, qsave
from os import makedirs
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sys
import time
from qutip import *

# ==========================================
# 0. FUNCIONES AUXILIARES
# ==========================================
def phi_n(n, x):
    return 1/math.sqrt(2**n*math.factorial(n))*math.pi**(-0.25)*np.exp(-np.array(x)**2/2)*special.eval_hermite(n, x)

def calcular_bloque_einsum(m, mp, rho_val, phi_p, D2, N2):
    B = np.einsum('ni,ij,mj->nm', phi_p, np.sqrt(rho_val.astype('complex128')), phi_p, optimize=True) * D2
    B2=np.einsum('ab,cd->abcd', B, B, optimize=True)
    return m, mp, B2.reshape(N2, N2)

k=0.5
# ==========================================
# 1. PARÁMETROS FÍSICOS Y DE LA CUADRÍCULA
# ==========================================
q_max = 10 
N_points = 3000
Delta = 2 * q_max / (N_points - 1)

   # Tu parámetro k
#betas = np.logspace(-0.2, 0.2, 50)    # Beta para la gráfica y matriz densidad
betas=[1]
N_Q = 2
NQ2 = N_Q**2  # NQ2 = 4 (Dimensión de la matriz)


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
    k_sym: k
}

# Sustituimos todo y preparamos lambdify
H_sub = H_sym.subs({E: Afun, T: Cfun}).subs(parametros)
H_tuple = tuple(tuple(H_sub[i, j] for j in range(NQ2)) for i in range(NQ2))
H_func = lambdify((q, p), H_tuple, modules='numpy')

# ==========================================
# 3. EVALUACIÓN Y EXPONENCIAL NUMÉRICA
# ==========================================
start_time=time.time()
res = H_func(q_grid, p_grid)
H_grid = np.empty((NQ2, NQ2, N_points, N_points))
for i in range(NQ2):
    for j in range(NQ2):
        H_grid[i, j] = res[i][j]
del res


# Se calcula phi_evals ANTES del bucle beta, ahora que phi_n ya existe

# Diagonalización vectorizada a lo largo de la cuadrícula
evals, evecs = np.linalg.eigh(np.moveaxis(H_grid, [0, 1], [-2, -1])) 
del H_grid

# ==========================================
# 4. BUCLE PARA CADA TEMPERATURA (BETA)
# ==========================================
for beta in betas:
    
    exp_evals = np.exp(-beta * evals) 
    expH_grid = np.moveaxis((evecs * exp_evals[..., None, :]) @ np.swapaxes(evecs.conj(), -1, -2), [-2, -1], [0, 1])
    del exp_evals

    tr_grid = np.trace(expH_grid, axis1=0, axis2=1).real # .real por seguridad numérica
    Z = np.sum(tr_grid) * Delta**2

    
    # Matriz densidad local
    rho_xi = expH_grid / Z
    del expH_grid, tr_grid
    for N_c in range(3,16):
        start_time=time.time()
        
        phi_evals = np.array([[phi_n(n, x) for x in qp_list] for n in range(N_c)])
            
        Nc2 = N_c**2
        dim = Nc2 * NQ2
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
            
        rho_qobj = Qobj(rho_HCE)
        rho_qobj.dims = [[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]]

        a= destroy(N_c)
        a_d=a.dag()
        Q = 1/np.sqrt(2)*(a+a_d)
        Q2op1 = tensor(qeye(N_Q), qeye(N_Q), Q*Q, qeye(N_c))
        Q2op2 = tensor(qeye(N_Q), qeye(N_Q), qeye(N_c), Q*Q)
        QPop = tensor(qeye(N_Q), qeye(N_Q), Q, Q)
        op10 = tensor(sigmax(), qeye(N_Q), qeye(N_c), qeye(N_c))
        op20 = tensor(sigmay(), qeye(N_Q), qeye(N_c), qeye(N_c))
        op30 = tensor(sigmaz(), qeye(N_Q), qeye(N_c), qeye(N_c))
        op01 = tensor(qeye(N_Q), sigmax(), qeye(N_c), qeye(N_c))
        op02 = tensor(qeye(N_Q), sigmay(), qeye(N_c), qeye(N_c))
        op03 = tensor(qeye(N_Q), sigmaz(), qeye(N_c), qeye(N_c))
        op11 = tensor(sigmax(), sigmax(), qeye(N_c), qeye(N_c))
        op12 = tensor(sigmax(), sigmay(), qeye(N_c), qeye(N_c))
        op13 = tensor(sigmax(), sigmaz(), qeye(N_c), qeye(N_c))
        op21 = tensor(sigmay(), sigmax(), qeye(N_c), qeye(N_c))
        op22 = tensor(sigmay(), sigmay(), qeye(N_c), qeye(N_c))
        op23 = tensor(sigmay(), sigmaz(), qeye(N_c), qeye(N_c))
        op31 = tensor(sigmaz(), sigmax(), qeye(N_c), qeye(N_c))
        op32 = tensor(sigmaz(), sigmay(), qeye(N_c), qeye(N_c))
        op33 = tensor(sigmaz(), sigmaz(), qeye(N_c), qeye(N_c))
        rhos_list = [op10, op20, op30, op01, op02, op03, op11, op12, op13, op21, op22, op23, op31, op32, op33]
        ops_qutip = [Q2op1, Q2op2, QPop] + rhos_list

        valores_qutip = expect(ops_qutip, rho_qobj)
        # Convertimos los valores numéricos a texto para poder unirlos con join
        valores_str = [str(v) for v in valores_qutip]

        tiempo_ejec = time.time() - start_time

        # qsave(rho_qobj, f"molecular_model/HCEs_no_diag/k_{k_val}/HCE_{beta:.4f}")

        with open(f"molecular_model/dim_test/k_{k}/dim_analysis_{k}.csv", "a") as f:
            # Añadida la coma (",") justo antes de imprimir el tiempo
            f.write(f"{N_c}," + ",".join(valores_str) + f",{tiempo_ejec}\n")

