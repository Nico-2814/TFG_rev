from sympy import *
import numpy as np
import pickle



q, p, E, T, b = symbols("q p E T b")
H=0.5*Matrix([[p**2+q**2+2*E*(1+cos(q))+2*T*(1-cos(q)), sin(q)*(E-T), sin(q)*(E-T), 0], [sin(q)*(E-T),p**2+q**2+2*E+2*T,0,sin(q)*(E-T)], [sin(q)*(E-T),0,p**2+q**2+2*E+2*T,sin(q)*(E-T)], [0,sin(q)*(E-T),sin(q)*(E-T),p**2+q**2+2*E*(1-cos(q))+2*T*(1+cos(q))]])


b=1

expH = (simplify(exp(-b*H))).subs({E: 1/(1+q**2), T: 1/(1+q**2)+1+0.1*q**2, b: 1})
#Z=2*pi/b*(exp(-b*E)+exp(-b*T))**2

fun = lambdify((q,p), exp(-0.5*p**2-0.5*q**2-2/(1+q**2))*(exp(-1-0.1*q**2)+1)**2, modules = 'numpy')
tr=simplify(trace(expH))
print(tr)
q_max=10
N_points=3000
Delta = 2*q_max/(N_points-1)
qp_list=np.linspace(-q_max,q_max,N_points)
tr_func = lambdify((q,p), tr,  modules='numpy')
q_grid, p_grid = np.meshgrid(qp_list, qp_list, indexing='ij')
tr_eval = tr_func(q_grid, p_grid)

Z = np.einsum('ij->', tr_eval)*Delta**2
print("Z: ", Z)

rho_xi = expH/Z

with open('exponential.pkl', 'wb') as f:
    pickle.dump(rho_xi, f)
