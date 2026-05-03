import numpy as np
import qutip as qt
import scipy.optimize as opt
import os


N_c = 5 #Number of classical states for each oscillator
S = 0.5 #Spin
N_Q = int(2*S+1)
Max_ops = 10 #máximo número de operadores a buscar

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

Q2 = Q*Q
P2 = P*P
QP = Q*P

Q3 = Q2*Q
P3 = P2*P
Q2P = Q2*P
QP2 = Q*P2

H = qt.Qobj(np.load("hamiltonian_5_2.npy", allow_pickle=True))
rho0 = qt.Qobj(np.load("HCE_5.npy", allow_pickle=True))
H.dims = rho0.dims = [[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]]

print("Precalculando tensores base")

basis = [
    # Orden 3
    qt.tensor(I_Q, I_Q, Q3 * Pi_Q), qt.tensor(I_Q, I_Q, Q2P * Pi_Q), qt.tensor(I_Q, I_Q, QP2 * Pi_Q), qt.tensor(I_Q, I_Q, P3 * Pi_Q),
    qt.tensor(I_Q, I_Q, Q3 * Pi_P), qt.tensor(I_Q, I_Q, Q2P * Pi_P), qt.tensor(I_Q, I_Q, QP2 * Pi_P), qt.tensor(I_Q, I_Q, P3 * Pi_P),
    # Orden 2
    qt.tensor(I_Q, I_Q, Q2 * Pi_Q), qt.tensor(I_Q, I_Q, QP * Pi_Q), qt.tensor(I_Q, I_Q, P2 * Pi_Q), 
    qt.tensor(I_Q, I_Q, Q2 * Pi_P), qt.tensor(I_Q, I_Q, QP * Pi_P), qt.tensor(I_Q, I_Q, P2 * Pi_P),
    # Orden 1
    qt.tensor(I_Q, I_Q, Q * Pi_Q), qt.tensor(I_Q, I_Q, P * Pi_Q), 
    qt.tensor(I_Q, I_Q, Q * Pi_P), qt.tensor(I_Q, I_Q, P * Pi_P)
]

spin_ops = [
    (I_Q, I_Q), (qt.sigmax(), I_Q), (qt.sigmay(), I_Q), (qt.sigmaz(), I_Q),
    (I_Q, qt.sigmax()), (I_Q, qt.sigmay()), (I_Q, qt.sigmaz()),
    (qt.sigmax(), qt.sigmax()), (qt.sigmax(), qt.sigmay()), (qt.sigmax(), qt.sigmaz()),
    (qt.sigmay(), qt.sigmax()), (qt.sigmay(), qt.sigmay()), (qt.sigmay(), qt.sigmaz()),
    (qt.sigmaz(), qt.sigmax()), (qt.sigmaz(), qt.sigmay()), (qt.sigmaz(), qt.sigmaz())
]

for op1, op2 in spin_ops:
    # Orden 3
    basis.append(qt.tensor(op1, op2, Q3))
    basis.append(qt.tensor(op1, op2, Q2P))
    basis.append(qt.tensor(op1, op2, QP2))
    basis.append(qt.tensor(op1, op2, P3))
    # Orden 2
    basis.append(qt.tensor(op1, op2, Q2))
    basis.append(qt.tensor(op1, op2, QP))
    basis.append(qt.tensor(op1, op2, P2))
    # Orden 1
    basis.append(qt.tensor(op1, op2, Q))
    basis.append(qt.tensor(op1, op2, P))

num_basis = len(basis) 
dyn_part = -1j * (H * rho0 - rho0 * H) #l conmutador de la ecuación de Lindblad
iter_count = 0

def cost(p):
    num_current_ops = len(p) // (2 * num_basis)
    diss_tot = 0
    
    for m in range(num_current_ops):
        start = m * 2 * num_basis
        c = p[start : start + num_basis] + 1j * p[start + num_basis : start + 2*num_basis]
        K = sum(coeff * matrix for coeff, matrix in zip(c, basis))
        K_dag = K.dag()
        K_dag_K = K_dag * K
        diss_tot += (K * rho0 * K_dag) - 0.5 * (K_dag_K * rho0 + rho0 * K_dag_K) #Toda esta sección construye los operadores de salto y calcula el término disipativo.
        
    var_mat = dyn_part + diss_tot
    return var_mat.norm('fro')

def printer(xk):
    global iter_count
    iter_count += 1
    print(f"  Iteración {iter_count} | Distancia a cero: {(cost(xk)):.6f}", flush=True)

# Esta parte carga si ya tengo calculados operadores de salto de antes para tomarlos de base, pero en general no se usa.
p_actual = np.random.rand(num_basis * 2 * 2) * 0.0001
'''
try:
    with open("jump_ops/K_params_2ops.txt", "r") as f:
        lineas = [linea for linea in f.readlines() if 'j' in linea]
        
    if len(lineas) == num_basis * 2:
        for idx, linea in enumerate(lineas):
            partes = linea.split('+')
            real = float(partes[0].strip())
            imag = float(partes[1].replace('j', '').strip())
            if idx < num_basis:
                p_actual[idx] = real; p_actual[idx + num_basis] = imag
            elif idx < 2 * num_basis:
                i_k2 = idx - num_basis
                p_actual[2 * num_basis + i_k2] = real; p_actual[3 * num_basis + i_k2] = imag
        print("Archivo compatible detectado. Parámetros cargados con éxito.")
    else:
        print(f"El archivo antiguo tiene {len(lineas)} parámetros, pero la base cúbica requiere {num_basis*2}.")
        print("Ignorando archivo y empezando desde cero con ruido.")
except FileNotFoundError:
    print("No se encontró 'K_params_2ops.txt', empezando desde cero.")
'''
# Esta es la parte de optimización, añadiendo cada vez un operador de salto
prev_best_cost = float('inf')

for ops in range(2, Max_ops + 1):
    print(f"\n==================================================")
    print(f"🚀 INICIANDO FASE: {ops} OPERADORES DE SALTO (ORDEN 3)")
    print(f"==================================================")
    iter_count = 0
    
    result = opt.minimize(cost, p_actual, method='L-BFGS-B', callback=printer, options={'maxiter': 300})
    
    current_best_cost = result.fun
    print(f"✅ Fase {ops} completada. Coste final: {current_best_cost:.6f}")
    
    #Analiza si merece la pena añadir más operadores
    if ops > 2:
        mejora = prev_best_cost - current_best_cost
        print(f"  -> Mejora conseguida por el operador {ops}: {mejora:.6f}")
        
        if mejora < 0.001:
            print(f"La mejora ({mejora:.6f}) ha caído por debajo del umbral de 0.001. Deteniendo la búsqueda.")
            p_actual = result.x
            break

    # Guardamos el mejor coste y actualizamos los parámetros para la siguiente ronda
    prev_best_cost = current_best_cost
    p_actual = result.x
    
    # Preparar la siguiente fase añadiendo ruido para el nuevo operador 
    if ops < Max_ops:
        nuevo_operador_ruido = np.random.rand(num_basis * 2) * 0.0001
        p_actual = np.concatenate((p_actual, nuevo_operador_ruido))

#guardado de resultados
num_final_ops = len(p_actual) // (2 * num_basis)

print(f"\nGuardando los {num_final_ops} operadores definitivos")
os.makedirs("jump_ops", exist_ok=True)

with open("jump_ops/K_params_final_cubeeee.txt", "w") as f:
    for m in range(num_final_ops):
        f.write(f"=== Parametros K{m+1} ===\n")
        start = m * 2 * num_basis
        c_opt = p_actual[start : start + num_basis] + 1j * p_actual[start + num_basis : start + 2*num_basis]
        
        K_opt = sum(coeff * matrix for coeff, matrix in zip(c_opt, basis))
        qt.qsave(K_opt, f"jump_ops/K{m+1}cubeeee")
        
        for i in range(num_basis):
            f.write(f"{c_opt[i].real} + {c_opt[i].imag}j\n")
        f.write("\n")

print(f"Todos los {num_final_ops} operadores guardados con éxito")