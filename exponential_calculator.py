from sympy import *
import numpy as np
import pickle
import matplotlib.pyplot as plt

# 1. Declaramos todos los símbolos correctamente
q, p, A, B, C, D, k, b, E, T = symbols("q p A B C D k b E T")

Afun = Piecewise(
    (A * (1 - exp(-B*q)), q>0),
    (A*(exp(B*q)-1), True)
) 
Cfun = C * exp(-D*q**2)

H = Matrix([
    [0.5*k*(p**2+q**2)+2*E, T, T, 0], 
    [T, 0.5*k*(p**2+q**2), 0, T], 
    [T, 0, 0.5*k*(p**2+q**2), T], 
    [0, T, T, 0.5*k*(p**N2+q**2)-2*E]
])

# 2. Definimos los valores numéricos de tus parámetros
k_val = 0.0
parametros = {
    A: 0.01,
    B: 1.6,
    C: 0.005,
    D: 1.0,
    k: k_val,
    b: 1.0
}

# Sustituimos E, T y luego los parámetros numéricos
H_sub = H.subs({E: Afun, T: Cfun}).subs(parametros)

# 3. Calculamos la exponencial de la MATRIZ usando .exp()
# (Nota: simplify() aquí puede tardar un poco dependiendo de la complejidad)
expH = simplify((-parametros[b] * H_sub).exp())
print("Exponencial calculada")
tr = simplify(trace(expH))
print("Traza calculada:\n", tr)

# 4. Configuración de la cuadrícula NumPy
q_max = 10
N_points = 300  # Reducido a 300 para probar rápido, sube a 3000 luego si tienes suficiente RAM
Delta = 2*q_max / (N_points - 1)
qp_list = np.linspace(-q_max, q_max, N_points)

# lambdify requiere que ya no queden letras (A, B, etc.), solo q y p
tr_func = lambdify((q, p), tr, modules='numpy')
q_grid, p_grid = np.meshgrid(qp_list, qp_list, indexing='ij')

# Evaluación numérica
tr_eval = tr_func(q_grid, p_grid)
Z = np.einsum('ij->', tr_eval) * Delta**2
print(f"Función de pPartición Z = {Z}")

# Matriz densidad (simbólica)
rho_xi = expH / Z

# Guardado
with open(f'molecular_model/exponential_k={k}.pkl', 'wb') as f:
    pickle.dump(rho_xi, f)

# 5. Gráfico corregido
plt.figure(figsize=(8, 6))

# Usamos q_grid y p_grid en lugar de X e Y
mapa = plt.pcolormesh(q_grid, p_grid, tr_eval, cmap='viridis', shading='auto')

plt.colorbar(mapa)
# Título con el valor de k
plt.title(rf'Tr$\left(e^{{-\beta*H(q,p)}}\right)$, $k={k_val}$')
# Etiquetas de ejes con la 'r' por fuera de las comillas
plt.xlabel(r'$q$')
plt.ylabel(r'$p$')

plt.savefig(f"molecular_model/trace_k={k_val}.png")
