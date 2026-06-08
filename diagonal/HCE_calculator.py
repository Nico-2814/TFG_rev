import numpy as np
from scipy import *
from sympy import *
import math
import pickle
from scipy import special

q_max=10
N_points=3000
Delta = 2*q_max/(N_points-1)
N_c=5
N_Q=2

def phi_n (n, x):
    return 1/math.sqrt(2**n*math.factorial(n))*math.pi**(-0.25)*np.exp(-np.array(x)**2/2)*special.eval_hermite(n, x)

qp_list=np.linspace(-q_max,q_max,N_points)
phi_evals = np.array([[phi_n(n,x) for x in qp_list] for n in range(N_c)])

q, p, E, T, b = symbols("q p E T b")
H=0.5*Matrix([[p**2+q**2+2*E*(1+cos(q))+2*T*(1-cos(q)), sin(q)*(E-T), sin(q)*(E-T), 0], [sin(q)*(E-T),p**2+q**2+2*E+2*T,0,sin(q)*(E-T)], [sin(q)*(E-T),0,p**2+q**2+2*E+2*T,sin(q)*(E-T)], [0,sin(q)*(E-T),sin(q)*(E-T),p**2+q**2+2*E*(1-cos(q))+2*T*(1+cos(q))]])
Z=3.381726919890751 #se calcula con exponential calculator

with open('exponential.pkl', 'rb') as f:
    rho_xi = pickle.load(f)
exp_func = lambdify((q,p), rho_xi,  modules='numpy')
q_grid, p_grid = np.meshgrid(qp_list, qp_list, indexing='ij')
exp_eval = exp_func(q_grid, p_grid)

H_subs=H.subs({E: 1/(1+q**2), T: 1/(1+q**2)+1+0.1*q**2, b: 1})
H_eval=np.zeros((4,4,N_points, N_points))
for i in range(4):
    for j in range(4):
        func_cell = lambdify((q,p), H_subs[i, j], modules="numpy")
        H_eval[i, j] = func_cell(q_grid, p_grid)


sum=0
L=math.log(Z)
I_amp=np.eye(4)[:, :, np.newaxis, np.newaxis]
sum=np.einsum('mnij, nmij ->', exp_eval, H_eval+I_amp*L)
print("Entropía: ", Delta*Delta*sum)



