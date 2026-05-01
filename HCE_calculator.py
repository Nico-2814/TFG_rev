import numpy as np
from sympy import *
import math
from scipy import special
from qutip import qsave, Qobj

#Propiedades de la red del espacio de fases clásico
q_max = 10 
N_points = 3000 
Delta = 2*q_max/(N_points-1)
N_c=10
N_Q=2

Nc2=N_c**2
NQ2=N_Q**2

def phi_n(n, x):
    return 1/math.sqrt(2**n*math.factorial(n))*math.pi**(-0.25)*np.exp(-np.array(x)**2/2)*special.eval_hermite(n, x)

qp_list = np.linspace(-q_max, q_max, N_points)
q_grid, p_grid = np.meshgrid(qp_list, qp_list, indexing='ij')
phi_evals = np.array([[phi_n(n,x) for x in qp_list] for n in range(N_c)])

q, p, E, T = symbols("q p E T")
H_sym = 0.5*Matrix([
    [p**2+q**2+2*E*(1+cos(q))+2*T*(1-cos(q)), sin(q)*(E-T), sin(q)*(E-T), 0], 
    [sin(q)*(E-T), p**2+q**2+2*E+2*T, 0, sin(q)*(E-T)], 
    [sin(q)*(E-T), 0, p**2+q**2+2*E+2*T, sin(q)*(E-T)], 
    [0, sin(q)*(E-T), sin(q)*(E-T), p**2+q**2+2*E*(1-cos(q))+2*T*(1+cos(q))]
])
H_sub = H_sym.subs({E: 1/(1+q**2), T: 1/(1+q**2)+1+0.1*q**2})
H_tuple = tuple(tuple(H_sub[i, j] for j in range(NQ2)) for i in range(NQ2))
H_func = lambdify((q, p), H_tuple, modules='numpy')
res = H_func(q_grid, p_grid)
H_grid = np.empty((4, 4, N_points, N_points)) #Evalúa el hamiltoniano en todos los puntos para calcular la integral de normalización
for i in range(4):
    for j in range(4):
        H_grid[i, j] = res[i][j]
evals, evecs = np.linalg.eigh(np.moveaxis(H_grid, [0, 1], [-2, -1])) 
del res
del H_tuple
del H_func

betas = np.logspace(-5, 1, 50)
Delta4 = Delta**4
dim = Nc2 * NQ2

for beta in betas:
    exp_evals = np.exp(-beta * evals) 
    expH_grid = np.moveaxis((evecs * exp_evals[..., None, :]) @ np.swapaxes(evecs.conj(), -1, -2), [-2, -1], [0, 1])
    del exp_evals

    tr_grid = np.trace(expH_grid, axis1=0, axis2=1)
    Z = np.sum(tr_grid) * Delta**2
    del tr_grid
    rho_xi = expH_grid / Z
    del expH_grid
    
    rho_HCE = np.zeros((dim, dim), dtype='complex128')
    
    for m in range(NQ2):
        for mp in range(NQ2):
            root_exp = np.sqrt(rho_xi[m, mp].astype(np.complex128))
            A = phi_evals @ root_exp @ phi_evals.T 
            B = np.einsum('nm,op->nmop', A, A) * Delta4
            del A
            B_2d = B.reshape(Nc2, Nc2)
            del B
            rho_HCE[Nc2*m:Nc2*(m+1), Nc2*mp:Nc2*(mp+1)] = B_2d

    rho_qobj = Qobj(rho_HCE)
    rho_qobj.dims = [[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]]
    qsave(rho_qobj, f"HCEs/HCE_{beta}_{N_c}")