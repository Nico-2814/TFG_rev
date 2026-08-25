import os
import time
import jax
import jax.numpy as jnp
import qutip as qt
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize as opt
from functools import partial

# ==============================================================================
# CONFIGURACIÓN Y HARDWARE
# ==============================================================================
print("\n--- REPORTE DE HARDWARE ---")
dispositivos = jax.devices()
print(f"Dispositivos encontrados: {dispositivos}")

device_str = str(dispositivos[0]).lower()
if 'cuda' in device_str or 'gpu' in device_str:
    print(" GPU detectada correctamente. ")
elif 'cpu' in device_str:
    print(" ¡OJO! Estás en CPU. ")

os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='

# Configuración de rutas (Ajustar según entorno)
input_path = '/kaggle/input/datasets/nico880745/molecular-model-tfg-fis/'
output_path = '/kaggle/working/'
k = 0.5
lambda_energia = 1.0

# Creación de directorios
directorios = ["results/Ks", "results/steadies", "results/evals", "results/temps"]
for d in directorios:
    os.makedirs(os.path.join(output_path, d), exist_ok=True)

# Parámetros Cuánticos
N_c = 10 
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

# Carga de H y rho_0
H_np = np.load(input_path+f"hamiltonian_k_{k}_2_10.npy")
rho_0np = np.array((qt.qload(input_path+f"HCE_1.0000_10")).full())
rho_0np = rho_0np / np.trace(rho_0np)

H_jax = jnp.array(H_np, dtype=jnp.complex128)
rho0_jax = jnp.array(rho_0np, dtype=jnp.complex128)

start_time_total=time.time()
# ==============================================================================
# CONSTRUCCIÓN DE LA BASE (FÍSICA RIGUROSA)
# ==============================================================================
def basis_creator(max_g):
    print(f"\nPrecalculando tensores base para orden {max_g}...")
    basis_pi = []
    ordenes_pi = []             
    
    monomios_fase_total = [] 
    ordenes_monomios = []       
    
    for order in range(max_g, -1, -1): 
        monomios_orden = []
        for exp_Q in range(order, -1, -1):
            exp_P = order - exp_Q
            monomios_orden.append(Q**exp_Q * P**exp_P)
            ordenes_monomios.append(order) 
            
        monomios_fase_total.extend(monomios_orden)
        
        # 1. Base Pi
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(I_Q, I_Q, monomio))
            ordenes_pi.append(order)
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(I_Q, I_Q, monomio * Pi_Q))
            ordenes_pi.append(order)
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(I_Q, I_Q, monomio * Pi_P))
            ordenes_pi.append(order)

    # 2. Operadores de Espín
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
        basis_spin.append(op_basis)
            
    return (
        basis_pi, 
        basis_spin, 
        len(monomios_fase_total), 
        np.array(ordenes_pi, dtype=np.float32), 
        np.array(ordenes_monomios, dtype=np.float32)
    )

def normalizar_base_matrices(basis):
    xp = jnp if isinstance(basis, jnp.ndarray) else np
    normas = xp.linalg.norm(basis, axis=(-2, -1), keepdims=True)
    normas = xp.where(normas == 0, 1.0, normas)
    return basis / normas

def normalizar_base_qutip(basis_list):
    if isinstance(basis_list[0], list): 
        return [[op / max(np.linalg.norm(op.full()), 1e-12) for op in ops] for ops in basis_list]
    else: 
        return [op / max(np.linalg.norm(op.full()), 1e-12) for op in basis_list]


# ==============================================================================
# DEFINICIÓN DEL COSTE 
# ==============================================================================

import jax.numpy as jnp
import numpy as np
import qutip as qt

dim = rho_0np.shape[0]
# ==============================================================================
# GENERACIÓN DE PERTURBACIONES CON QUTIP
# ==============================================================================
def generar_deltas_qutip(dim, num_deltas=3):
    deltas = []
    for _ in range(num_deltas):
        H_qt = qt.rand_herm(dim)
    
        H_tr = H_qt - (H_qt.tr() / dim) * qt.qeye(dim)
        
        mat = H_tr.full()
        mat = mat / np.linalg.norm(mat)
        deltas.append(mat)
        
    return jnp.array(deltas)


# ==============================================================================
# CREADOR DE FUNCIONES DE OPTIMIZACIÓN (JAX + GPU)
# ==============================================================================
def crear_funciones_optimizacion(
    basis_Pi_jax, 
    basis_spin_jax, 
    pesos_orden_Pi_jax, 
    pesos_orden_Sigma_jax,
    NUM_K_PI, 
    NUM_K_SIGMA=1, 
    gamma_L2=1e-3,
    epsilon=0.05,
    peso_atractor=10.0
):
    num_basis_Pi = basis_Pi_jax.shape[0]
    num_spin_ops = basis_spin_jax.shape[0] 
    num_monomios = basis_spin_jax.shape[1]
    deltas_jax = generar_deltas_qutip(dim, num_deltas=3)

    @jax.jit
    def cost_jax(p, deltas_jax):
        p_Pi_end = NUM_K_PI * 2 * num_basis_Pi
        p_Pi = p[:p_Pi_end]
        p_sigma = p[p_Pi_end:]
        
        c_ops = []
        penalizacion_L2 = 0.0
        
        # ----------------------------------------------------------------------
        # A. CANALES PI + REGULARIZACIÓN L2 POR ORDEN (g^2)
        # ----------------------------------------------------------------------
        for m in range(NUM_K_PI):
            start = m * 2 * num_basis_Pi
            params_m_Pi = p_Pi[start : start + 2*num_basis_Pi]
            
            c_real = params_m_Pi[:num_basis_Pi]
            c_imag = params_m_Pi[num_basis_Pi:]
            c = c_real + 1j * c_imag
            
            c_ops.append(jnp.sum(c[:, None, None] * basis_Pi_jax, axis=0))
            
            # Penalización L2 cuadrática con el orden físico
            penalizacion_L2 += gamma_L2 * jnp.sum((pesos_orden_Pi_jax**2) * (c_real**2 + c_imag**2))
            
        # ----------------------------------------------------------------------
        # B. CANALES SIGMA + REGULARIZACIÓN L2 POR ORDEN (g^2)
        # ----------------------------------------------------------------------
        params_per_sigma = 2 * num_monomios + 2 * num_spin_ops
        for m in range(NUM_K_SIGMA):
            start = m * params_per_sigma
            
            d_real = p_sigma[start : start + num_monomios]
            d_imag = p_sigma[start + num_monomios : start + 2*num_monomios]
            d_cpx = d_real + 1j * d_imag
            
            c_start = start + 2*num_monomios
            c_cpx = p_sigma[c_start : c_start + num_spin_ops] + 1j * p_sigma[c_start + num_spin_ops : c_start + 2*num_spin_ops]
            
            tensor_coefs = c_cpx[:, None] * d_cpx[None, :]
            V_sigma = jnp.sum(tensor_coefs[:, :, None, None] * basis_spin_jax, axis=(0, 1))
            c_ops.append(V_sigma)
            
            penalizacion_L2 += gamma_L2 * jnp.sum((pesos_orden_Sigma_jax**2) * (d_real**2 + d_imag**2))
            
        # ----------------------------------------------------------------------
        # C. ACCIÓN DEL LIOUVILLIANO L(rho)
        # ----------------------------------------------------------------------
        def aplicar_liouvilliano(rho):
            L_rho = -1j * (H_jax @ rho - rho @ H_jax)
            for K in c_ops:
                K_dag = K.conj().T
                K_tail = K_dag @ K
                L_rho += (K @ rho @ K_dag) - 0.5 * (K_tail @ rho + rho @ K_tail)
            return L_rho

        # 1. Coste Velocidad
        L_rho0 = aplicar_liouvilliano(rho0_jax)
        coste_velocidad = jnp.sum(jnp.abs(L_rho0)**2)
        
        # 2. Evaluación sobre estados perturbados usando VMAP (AHORRA MUCHÍSIMA MEMORIA)
        def calcular_flujo(d_rho):
            rho_pert = rho0_jax + epsilon * d_rho
            L_rho_pert = aplicar_liouvilliano(rho_pert)
            
            flujo = jnp.real(jnp.trace((rho_pert - rho0_jax).conj().T @ L_rho_pert))
            return jnp.maximum(0.0, flujo + 1e-3)
            
        # vmap compila 'calcular_flujo' una sola vez y lo ejecuta en paralelo para todas las deltas
        flujos_penalizados = jax.vmap(calcular_flujo)(deltas_jax)
        penalizacion_atractor = jnp.sum(flujos_penalizados)

        coste_total = coste_velocidad + (peso_atractor * penalizacion_atractor) 
        return coste_total * 100.0
        
    cost_and_grad = jax.value_and_grad(cost_jax)

    def scipy_objective(p):
        val, grad = cost_and_grad(jnp.array(p), deltas_jax)
        return float(val), np.array(grad, dtype=np.float64)
        
    return scipy_objective, cost_and_grad

# ==============================================================================
# MOTOR DE OPTIMIZACIÓN HÍBRIDO (ADAM + L-BFGS-B)
# ==============================================================================
def generar_deltas_jax(key, dim, num_deltas=5):
    keys = jax.random.split(key, num_deltas)
    
    def _gen_single(k):
        k1, k2 = jax.random.split(k)
        M = jax.random.normal(k1, (dim, dim)) + 1j * jax.random.normal(k2, (dim, dim))
        H = 0.5 * (M + M.conj().T)
        H_tr = H - (jnp.trace(H) / dim) * jnp.eye(dim)
        return H_tr / jnp.linalg.norm(H_tr)

    return jax.vmap(_gen_single)(keys)


def construir_adam_estocastico_gpu(cost_and_grad_fn, dim, num_deltas=5):
    @jax.jit(static_argnames=['num_steps'])
    def ejecutar_adam_gpu(p_init, num_steps, rng_key, lr=1e-3):
        b1, b2, eps = 0.9, 0.999, 1e-8
        
        def step_fn(state, carry_input):
            p, m, v = state
            i, key = carry_input
            
            deltas_step = generar_deltas_jax(key, dim, num_deltas)
            
            # Evaluar coste y gradiente con las deltas del paso actual
            val, grad = cost_and_grad_fn(p, deltas_step)
            
            # Paso de Adam
            m_new = b1 * m + (1.0 - b1) * grad
            v_new = b2 * v + (1.0 - b2) * jnp.square(grad)
            
            step_num = i + 1
            m_hat = m_new / (1.0 - b1**step_num)
            v_hat = v_new / (1.0 - b2**step_num)
            
            p_new = p - lr * m_hat / (jnp.sqrt(v_hat) + eps)
            return (p_new, m_new, v_new), val

        init_state = (p_init, jnp.zeros_like(p_init), jnp.zeros_like(p_init))
        
        # Generar sub-claves aleatorias para cada paso del loop
        keys = jax.random.split(rng_key, num_steps)
        steps = jnp.arange(num_steps)
        
        final_state, history = jax.lax.scan(step_fn, init_state, (steps, keys))
        return final_state[0], history
        
    return ejecutar_adam_gpu

def optimizar_hibrido(p_init, scipy_objective, adam_fn, iter_adam=200, iter_bfgs=1000, lr=1e-3):
    jax.clear_caches()
    """Fase 1: Escaneo rápido con Adam | Fase 2: Ajuste fino con L-BFGS-B"""
    p_jax = jnp.array(p_init)
    
    # 1. Adam
    if iter_adam > 0:
        p_opt_adam, coste_adam = adam_fn(p_jax, iter_adam, jax.random.PRNGKey(42), lr)
        print("     Adam completado")
        p_init_bfgs = np.array(p_opt_adam)
    else:
        p_init_bfgs = p_init
        
    # 2. L-BFGS-B
    res = opt.minimize(scipy_objective, p_init_bfgs, method='L-BFGS-B', jac=True, options={'maxiter': iter_bfgs})
    print(f"     L-BFGS-B: {res.nit} iteraciones")
    return res.x, res.fun


# ==============================================================================
# INYECCIÓN DINÁMICA DE CANALES (WARM STARTS)
# ==============================================================================
def inyectar_canal_pi(p_old, num_pi, NUM_K_PI_old):
    p_Pi_end_old = NUM_K_PI_old * 2 * num_pi
    p_Pi_old = p_old[:p_Pi_end_old]
    p_sigma_old = p_old[p_Pi_end_old:]
    
    nuevo_op_real = np.random.randn(num_pi) * 1e-6
    nuevo_op_imag = np.random.randn(num_pi) * 1e-6
    
    p_Pi_new = np.concatenate([p_Pi_old, nuevo_op_real, nuevo_op_imag])
    return np.concatenate([p_Pi_new, p_sigma_old])

def inyectar_canal_sigma(p_old, num_monomios):
    params_per_sigma = 2 * num_monomios + 30
    nuevo_op = np.random.randn(params_per_sigma) * 1e-6
    return np.concatenate([p_old, nuevo_op])

def inyectar_warm_start(p_old, num_pi_old, num_pi_new, num_monomios_old, num_monomios_new, NUM_K_PI, NUM_K_SIGMA=1):
    delta_pi = num_pi_new - num_pi_old
    delta_monomios = num_monomios_new - num_monomios_old
    p_new = []
    
    p_Pi_old = p_old[:NUM_K_PI * 2 * num_pi_old]
    for m in range(NUM_K_PI):
        start = m * 2 * num_pi_old
        real_old = p_Pi_old[start : start + num_pi_old]
        imag_old = p_Pi_old[start + num_pi_old : start + 2 * num_pi_old]
        
        real_new = np.concatenate([np.random.randn(delta_pi) * 1e-6, real_old])
        imag_new = np.concatenate([np.random.randn(delta_pi) * 1e-6, imag_old])
        p_new.extend(real_new)
        p_new.extend(imag_new)
        
    p_sigma_old = p_old[NUM_K_PI * 2 * num_pi_old:]
    params_per_sigma_old = 2 * num_monomios_old + 30
    
    for m in range(NUM_K_SIGMA):
        start = m * params_per_sigma_old
        d_real_old = p_sigma_old[start : start + num_monomios_old]
        d_imag_old = p_sigma_old[start + num_monomios_old : start + 2*num_monomios_old]
        
        c_start = start + 2*num_monomios_old
        c_real = p_sigma_old[c_start : c_start + 15]
        c_imag = p_sigma_old[c_start + 15 : c_start + 30]
        
        d_real_new = np.concatenate([np.random.randn(delta_monomios) * 1e-6, d_real_old])
        d_imag_new = np.concatenate([np.random.randn(delta_monomios) * 1e-6, d_imag_old])
        
        p_new.extend(d_real_new)
        p_new.extend(d_imag_new)
        p_new.extend(c_real)
        p_new.extend(c_imag)
        
    return np.array(p_new)

def inyectar_warm_start_cvxpy(coefs_pi_cpx, coefs_sigma_d_cpx, coefs_sigma_c_cpx, 
                              NUM_K_PI, NUM_K_SIGMA, num_basis_Pi, num_monomios, num_spin_ops=15):
    p_new = []
    
    for m in range(NUM_K_PI):
        if m < len(coefs_pi_cpx):
            c_old = np.array(coefs_pi_cpx[m])
            c = np.zeros(num_basis_Pi, dtype=complex)
            
            if len(c_old) <= num_basis_Pi:
                c[-len(c_old):] = c_old
                c[:-len(c_old)] = np.random.randn(num_basis_Pi - len(c_old)) * 1e-6 + 1j * np.random.randn(num_basis_Pi - len(c_old)) * 1e-6
            else:
                c = c_old[:num_basis_Pi]
        else:
            c = np.random.randn(num_basis_Pi) * 1e-6 + 1j * np.random.randn(num_basis_Pi) * 1e-6
            
        p_new.extend(c.real)
        p_new.extend(c.imag)
        
    for m in range(NUM_K_SIGMA):
        if m < len(coefs_sigma_d_cpx):
            d = np.array(coefs_sigma_d_cpx[m])
            c_spin = np.array(coefs_sigma_c_cpx[m])
        else:
            d = np.random.randn(num_monomios) * 1e-6 + 1j * np.random.randn(num_monomios) * 1e-6
            c_spin = np.random.randn(num_spin_ops) * 1e-6 + 1j * np.random.randn(num_spin_ops) * 1e-6
            
        p_new.extend(d.real)
        p_new.extend(d.imag)
        p_new.extend(c_spin.real)
        p_new.extend(c_spin.imag)
        
    return np.array(p_new, dtype=np.float64)

def guardar_parametros(filename, p, num_pi, num_monomios, NUM_K_PI, NUM_K_SIGMA=1):
    with open(filename, "w", encoding='utf-8') as f:
        f.write("=== OPERADORES DE DERIVADA (Pi) ===\n")
        p_Pi = p[:NUM_K_PI * 2 * num_pi]
        for m in range(NUM_K_PI):
            f.write(f"--- K_Pi_{m+1} ---\n")
            start = m * 2 * num_pi
            c = p_Pi[start : start + num_pi] + 1j * p_Pi[start + num_pi : start + 2*num_pi]
            for val in c: f.write(f"{val.real} + {val.imag}j\n")
            
        f.write("\n=== OPERADOR SIGMA ===\n")
        p_sigma = p[NUM_K_PI * 2 * num_pi:]
        params_per_sigma = 2 * num_monomios + 30
        for m in range(NUM_K_SIGMA):
            start = m * params_per_sigma
            d_real = p_sigma[start : start + num_monomios]
            d_imag = p_sigma[start + num_monomios : start + 2*num_monomios]
            d = d_real + 1j * d_imag
            
            c_start = start + 2*num_monomios
            c_real = p_sigma[c_start : c_start + 15]
            c_imag = p_sigma[c_start + 15 : c_start + 30]
            c = c_real + 1j * c_imag
            
            f.write(f"--- K_Sigma_{m+1}_Polinomio ---\n")
            for val in d: f.write(f"{val.real} + {val.imag}j\n")
            
            f.write(f"--- K_Sigma_{m+1}_Spins ---\n")
            for val in c: f.write(f"{val.real} + {val.imag}j\n")


import matplotlib.pyplot as plt

def cargar_operadores_K(filepath, basis_Pi, basis_spin):
    jump_ops = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("===")]
        
    blocks = {}
    current_name = ""
    for linea in lines:
        if linea.startswith("---"):
            current_name = linea.replace("-", "").strip()
            blocks[current_name] = []
        else:
            str_c = linea.replace(' ', '').replace('+-', '-')
            blocks[current_name].append(complex(str_c))
            
    for name, coefs in blocks.items():
        if "Pi" in name:
            K = 0 * basis_Pi[0]
            for c, op in zip(coefs, basis_Pi):
                K += c * op
            jump_ops.append(K)
            
    sigma_indices = set([name.split("_")[2] for name in blocks.keys() if "Sigma" in name])
    for idx in sigma_indices:
        d_coefs = blocks[f"K_Sigma_{idx}_Polinomio"]
        c_coefs = blocks[f"K_Sigma_{idx}_Spins"]
        
        V_sigma = 0 * basis_spin[0][0]
        for k, c_val in enumerate(c_coefs):
            for m, d_val in enumerate(d_coefs):
                V_sigma += c_val * d_val * basis_spin[k][m]
        jump_ops.append(V_sigma)
        
    return jump_ops

from scipy.sparse.linalg import LinearOperator, eigs

def calcular_autovalores_liouvilliano_gpu(H, K_ops, k=100):
        print(f"Calculando los {k} autovalores principales del Liouvilliano...")
        
        @jax.jit
        def apply_L_flat(rho_flat):
            rho = rho_flat.reshape((N, N))
            comm = -1j * (H @ rho - rho @ H)
            dissipator = jnp.zeros_like(rho)
            for K in K_ops:
                K_dag = K.conj().T
                K_tail = K_dag @ K
                dissipator += (K @ rho @ K_dag) - 0.5 * (K_tail @ rho + rho @ K_tail)
            return (comm + dissipator).flatten()

        def matvec(v):
            return np.array(apply_L_flat(jnp.array(v, dtype=jnp.complex128)))

        dim = N * N
        L_op = LinearOperator((dim, dim), matvec=matvec, dtype=np.complex128)
        
        evals, _ = eigs(L_op, k=k, which='LR', tol=1e-2, maxiter=1000)
        return evals

rho0qt = qt.Qobj(rho_0np, dims=[[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]])

# ==============================================================================
# EL BUCLE PRINCIPAL 
# ==============================================================================
NUM_K_PI = 4  
NUM_K_SIGMA = 10 
max_orden_final = 5

# --- INICIO EN ORDEN 2 ---
print("\n INICIANDO FASE 2 (ORDEN 2)")
basis_Pi, basis_spin, num_monomios_actual, pesos_pi, pesos_sigma = basis_creator(2)
num_basis_Pi_actual = len(basis_Pi)
num_spin_ops_actual = 15 

basis_Pi_jax = normalizar_base_matrices(jnp.array([op.full() for op in basis_Pi], dtype=jnp.complex128))
basis_spin_jax = normalizar_base_matrices(jnp.array([[op.full() for op in ops] for ops in basis_spin], dtype=jnp.complex128))
pesos_pi_jax=jnp.array(pesos_pi)
pesos_sigma_jax=jnp.array(pesos_sigma)

scipy_objective, cost_and_grad = crear_funciones_optimizacion(basis_Pi_jax, basis_spin_jax, pesos_pi_jax, pesos_sigma_jax, NUM_K_PI, NUM_K_SIGMA)
adam_gpu = construir_adam_estocastico_gpu(cost_and_grad, dim)

# =================================================================
# CARGA DEL WARM START DE CVXPY
# =================================================================
print("Cargando solución de CVXPY como Warm Start...")
datos_cvxpy = np.load(input_path + f"1_{N_c}.npz") # Ajusta tu ruta
coefs_pi_cvxpy = datos_cvxpy['coefs_pi']
K_ops_spin_raw = datos_cvxpy['coefs_spin']

coefs_sig_d_cvxpy = []
coefs_sig_c_cvxpy = []

for m in range(len(K_ops_spin_raw)):
    num_monomios_cvxpy = len(K_ops_spin_raw[m]) // num_spin_ops_actual
    matriz_canal = K_ops_spin_raw[m].reshape((num_spin_ops_actual, num_monomios_cvxpy))
    U, S, Vh = np.linalg.svd(matriz_canal, full_matrices=False)
    
    c_spin = np.sqrt(S[0]) * U[:, 0]
    d_espacial_cvxpy = np.sqrt(S[0]) * Vh[0, :]
    d_espacial = np.zeros(num_monomios_actual, dtype=complex)
    d_espacial[-num_monomios_cvxpy:] = d_espacial_cvxpy
    
    # Los nuevos órdenes se inicializan con un mínimo de ruido cuántico
    d_espacial[:-num_monomios_cvxpy] = np.random.randn(num_monomios_actual - num_monomios_cvxpy) * 1e-6
    
    coefs_sig_c_cvxpy.append(c_spin)
    coefs_sig_d_cvxpy.append(d_espacial)
# ----------------------------------------------

p_warm_start = inyectar_warm_start_cvxpy(
    coefs_pi_cvxpy, 
    coefs_sig_d_cvxpy, 
    coefs_sig_c_cvxpy, 
    NUM_K_PI, 
    NUM_K_SIGMA, 
    num_basis_Pi_actual, 
    num_monomios_actual, 
    num_spin_ops_actual
)

print(f"\n--- Maratonista WARM START ---")
p_final, coste_final = optimizar_hibrido(
    p_warm_start, scipy_objective, adam_gpu, iter_adam=200, iter_bfgs=1500
)

mejor_coste_global = coste_final
mejor_p_global = p_final
print(f"  -> Coste final alcanzado tras Warm Start: {coste_final}")
    

# =================================================================
print(" Iniciando competición de canales...")
NUM_K_PI_INIT=NUM_K_PI
NUM_K_SIGMA_INIT=NUM_K_SIGMA
mejora = True
while mejora and (NUM_K_PI<2*NUM_K_PI_INIT or NUM_K_SIGMA<2*NUM_K_SIGMA_INIT) and time.time()-start_time_total<41400:
    p_test_pi = inyectar_canal_pi(mejor_p_global, num_basis_Pi_actual, NUM_K_PI)
    scipy_obj_pi, grad_pi = crear_funciones_optimizacion(basis_Pi_jax, basis_spin_jax, pesos_pi_jax, pesos_sigma_jax, NUM_K_PI + 1, NUM_K_SIGMA)
    adam_pi = construir_adam_estocastico_gpu(grad_pi, dim)
    p_opt_pi, coste_pi = optimizar_hibrido(p_test_pi, scipy_obj_pi, adam_pi, iter_adam=200, iter_bfgs=50)
    
    p_test_sigma = inyectar_canal_sigma(mejor_p_global, num_monomios_actual)
    scipy_obj_sig, grad_sig = crear_funciones_optimizacion(basis_Pi_jax, basis_spin_jax, pesos_pi_jax, pesos_sigma_jax, NUM_K_PI, NUM_K_SIGMA + 1)
    adam_sig = construir_adam_estocastico_gpu(grad_sig, dim)
    p_opt_sig, coste_sigma = optimizar_hibrido(p_test_sigma, scipy_obj_sig, adam_sig, iter_adam=200, iter_bfgs=50)
    
    mejor_coste_prueba = min(coste_pi, coste_sigma)

    if mejor_coste_prueba < mejor_coste_global * 0.999:
        if coste_pi < coste_sigma and NUM_K_PI < 2 * NUM_K_PI_INIT:
            print(f"    GANA PI: Mejora detectada ({coste_pi}). Entrenando...")
            NUM_K_PI += 1
            p_final, c_final = optimizar_hibrido(p_opt_pi, scipy_obj_pi, adam_pi, iter_adam=100, iter_bfgs=1000)
            scipy_objective, cost_and_grad, adam_gpu = scipy_obj_pi, grad_pi, adam_pi 
        elif coste_sigma<=coste_pi and NUM_K_SIGMA < 2 * NUM_K_SIGMA_INIT:
            print(f"    GANA SIGMA: Mejora detectada ({coste_sigma}). Entrenando...")
            NUM_K_SIGMA += 1
            p_final, c_final = optimizar_hibrido(p_opt_sig, scipy_obj_sig, adam_sig, iter_adam=100, iter_bfgs=1000)
            scipy_objective, cost_and_grad, adam_gpu = scipy_obj_sig, grad_sig, adam_sig
        else:
            mejora=False
        mejor_coste_global = c_final
        mejor_p_global = p_final
    else:
        print("    Fin: Ningún canal extra mejora el 0.1%.")
        mejora = False
            
print(f" ORDEN 2 COMPLETADO. Mejor coste: {mejor_coste_global:.6f}")
guardar_parametros(output_path+"results/Ks/K_params_g2.txt", mejor_p_global, num_basis_Pi_actual, num_monomios_actual, NUM_K_PI, NUM_K_SIGMA)

print(f"\n==================================================")
print(f" POST-PROCESANDO ORDEN 2 (MATRIX-FREE GPU)")
print(f"==================================================")
    
basis_Pi_eval, basis_spin_eval, _, _, _ = basis_creator(2)
Ks = cargar_operadores_K(
    output_path+f"results/Ks/K_params_g2.txt", 
    normalizar_base_qutip(basis_Pi_eval), 
    normalizar_base_qutip(basis_spin_eval)
)
Ks_jax = [jnp.array(K.full(), dtype=jnp.complex128) for K in Ks]
N = H_jax.shape[0]

from jax.scipy.sparse.linalg import gmres

@jax.jit
def calcular_steady_state_gpu(H, K_ops):
    def apply_L(rho):
        comm = -1j * (H @ rho - rho @ H)
        dissipator = jnp.zeros_like(rho)
        for K in K_ops:
            K_dag = K.conj().T
            K_tail = K_dag @ K
            dissipator += (K @ rho @ K_dag) - 0.5 * (K_tail @ rho + rho @ K_tail)
        return comm + dissipator

    def matvec(rho_flat):
        rho = rho_flat.reshape((N, N))
        drho = apply_L(rho)
        drho_flat = drho.flatten()
        return drho_flat.at[0].set(jnp.trace(rho))

    rhs = jnp.zeros(N * N, dtype=jnp.complex128).at[0].set(1.0)
    x0 = (jnp.eye(N, dtype=jnp.complex128) / N).flatten()
    rho_steady_flat, info = gmres(matvec, rhs, x0=x0, tol=1e-8, restart=50, maxiter=100)
    
    return rho_steady_flat.reshape((N, N))

# =================================================================
    
print("Calculando estado estacionario en GPU con GMRES...")
start_time = time.time()
rho_steady = calcular_steady_state_gpu(H_jax, Ks_jax)
print(f" Estado estacionario calculado en {time.time() - start_time:.2f} s")

diff = rho_steady - rho0_jax
dist = jnp.sum(jnp.abs(diff)**2) / (N_c**2 * N_Q**2)
rho_qt = qt.Qobj(np.array(rho_steady), dims=[[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]])
    
print(f"-> Distancia final al estado objetivo: {dist:.8f}")
print(f"-> Fidelidad: {qt.fidelity(rho_qt, rho0qt):.6f}")
print(f"-> Distancia de traza: {qt.tracedist(rho_qt, rho0qt):.6f}")

evals = calcular_autovalores_liouvilliano_gpu(H_jax, Ks_jax, k=50) # Ajusta k según lo que tarde
np.save(output_path+f"results/evals/liouvillian_evals_g2.npy", evals)

qt.qsave(rho_qt, output_path+f"results/steadies/steady_g2")


# --- BUCLE DE WARM START (ESCALADA DE ÓRDENES) ---
for g in range(3, max_orden_final + 1):
    print(f"\n==================================================")
    print(f" SUBIENDO A ORDEN {g}")
    print(f"==================================================")
    
    num_pi_viejo = num_basis_Pi_actual
    num_monomios_viejo = num_monomios_actual
    
    basis_Pi, basis_spin, num_monomios_actual, pesos_pi, pesos_sigma = basis_creator(g)
    num_basis_Pi_actual = len(basis_Pi)
    
    basis_Pi_jax = normalizar_base_matrices(jnp.array([op.full() for op in basis_Pi], dtype=jnp.complex128))
    basis_spin_jax = normalizar_base_matrices(jnp.array([[op.full() for op in ops] for ops in basis_spin], dtype=jnp.complex128))
    pesos_pi_jax=jnp.array(pesos_pi)
    pesos_sigma_jax=jnp.array(pesos_sigma)
    
    scipy_objective, cost_and_grad = crear_funciones_optimizacion(basis_Pi_jax, basis_spin_jax, pesos_pi_jax, pesos_sigma_jax, NUM_K_PI, NUM_K_SIGMA)
    adam_gpu = construir_adam_estocastico_gpu(cost_and_grad, dim)
    
    p_actual = inyectar_warm_start(
        mejor_p_global, 
        num_pi_viejo, num_basis_Pi_actual, 
        num_monomios_viejo, num_monomios_actual, 
        NUM_K_PI, NUM_K_SIGMA
    )
    
    print("Optimizando (Adam + L-BFGS-B)")
    mejor_p_global, mejor_coste_actual = optimizar_hibrido(p_actual, scipy_objective, adam_gpu, iter_adam=500, iter_bfgs=1000)
    
    print(" Iniciando competición de canales")
    NUM_K_PI_INIT=NUM_K_PI
    NUM_K_SIGMA_INIT=NUM_K_SIGMA
    mejora = True
    while mejora and (NUM_K_PI<2*NUM_K_PI_INIT or NUM_K_SIGMA<2*NUM_K_SIGMA_INIT) and time.time()-start_time_total<41400:
        p_test_pi = inyectar_canal_pi(mejor_p_global, num_basis_Pi_actual, NUM_K_PI)
        scipy_obj_pi, grad_pi = crear_funciones_optimizacion(basis_Pi_jax, basis_spin_jax, pesos_pi_jax, pesos_sigma_jax, NUM_K_PI + 1, NUM_K_SIGMA)
        adam_pi = construir_adam_estocastico_gpu(grad_pi, dim)
        p_opt_pi, coste_pi = optimizar_hibrido(p_test_pi, scipy_obj_pi, adam_pi, iter_adam=200, iter_bfgs=50)
        
        p_test_sigma = inyectar_canal_sigma(mejor_p_global, num_monomios_actual)
        scipy_obj_sig, grad_sig = crear_funciones_optimizacion(basis_Pi_jax, basis_spin_jax, pesos_pi_jax, pesos_sigma_jax, NUM_K_PI, NUM_K_SIGMA + 1)
        adam_sig = construir_adam_estocastico_gpu(grad_sig, dim)
        p_opt_sig, coste_sigma = optimizar_hibrido(p_test_sigma, scipy_obj_sig, adam_sig, iter_adam=200, iter_bfgs=50)
        
        mejor_coste_prueba = min(coste_pi, coste_sigma)
    
        if mejor_coste_prueba < mejor_coste_global * 0.999:
            if coste_pi < coste_sigma and NUM_K_PI < 2 * NUM_K_PI_INIT:
                print(f"    GANA PI: Mejora detectada ({coste_pi}). Entrenando...")
                NUM_K_PI += 1
                p_final, c_final = optimizar_hibrido(p_opt_pi, scipy_obj_pi, adam_pi, iter_adam=100, iter_bfgs=1000)
                scipy_objective, cost_and_grad, adam_gpu = scipy_obj_pi, grad_pi, adam_pi 
            elif coste_sigma<=coste_pi and NUM_K_SIGMA < 2 * NUM_K_SIGMA_INIT:
                print(f"    GANA SIGMA: Mejora detectada ({coste_sigma}). Entrenando...")
                NUM_K_SIGMA += 1
                p_final, c_final = optimizar_hibrido(p_opt_sig, scipy_obj_sig, adam_sig, iter_adam=100, iter_bfgs=1000)
                scipy_objective, cost_and_grad, adam_gpu = scipy_obj_sig, grad_sig, adam_sig
            else:
                mejora=False
            mejor_coste_actual = c_final
            mejor_p_global = p_final
        else:
            print("    Fin: Ningún canal extra mejora el 0.1%.")
            mejora = False
        guardar_parametros(output_path+f"results/Ks/K_params_g{g}.txt", mejor_p_global, num_basis_Pi_actual, num_monomios_actual, NUM_K_PI, NUM_K_SIGMA)
        
        
    mejora_coste = mejor_coste_global - mejor_coste_actual        
    print(f" Orden {g} completado. Coste: {mejor_coste_actual} (Mejora: {mejora_coste})")
    guardar_parametros(output_path+f"results/Ks/K_params_g{g}.txt", mejor_p_global, num_basis_Pi_actual, num_monomios_actual, NUM_K_PI, NUM_K_SIGMA)
    
    if mejora_coste < 1e-3 * mejor_coste_global:
        print(f" Freno automático: El orden {g} apenas aporta mejora.")
                
    mejor_coste_global = mejor_coste_actual
    order=g

    print(f"\n==================================================")
    print(f" POST-PROCESANDO ORDEN {order} (MATRIX-FREE GPU)")
    print(f"==================================================")
    
    basis_Pi_eval, basis_spin_eval, _, _, _ = basis_creator(order)
    Ks = cargar_operadores_K(
        output_path+f"results/Ks/K_params_g{order}.txt", 
        normalizar_base_qutip(basis_Pi_eval), 
        normalizar_base_qutip(basis_spin_eval)
    )
    Ks_jax = [jnp.array(K.full(), dtype=jnp.complex128) for K in Ks]
    N = H_jax.shape[0]

    from jax.scipy.sparse.linalg import gmres

    # =================================================================
    
    print("Calculando estado estacionario en GPU con GMRES...")
    start_time = time.time()
    rho_steady = calcular_steady_state_gpu(H_jax, Ks_jax)
    print(f" Estado estacionario calculado en {time.time() - start_time:.2f} s")

    diff = rho_steady - rho0_jax
    dist = jnp.sum(jnp.abs(diff)**2) / (N_c**2 * N_Q**2)
    rho_qt = qt.Qobj(np.array(rho_steady), dims=[[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]])
    
    print(f"-> Distancia final al estado objetivo: {dist:.8f}")
    print(f"-> Fidelidad: {qt.fidelity(rho_qt, rho0qt):.6f}")
    print(f"-> Distancia de traza: {qt.tracedist(rho_qt, rho0qt):.6f}")

    evals = calcular_autovalores_liouvilliano_gpu(H_jax, Ks_jax, k=50) 
    np.save(output_path+f"results/evals/liouvillian_evals_g{order}.npy", evals)
    
    qt.qsave(rho_qt, output_path+f"results/steadies/steady_g{order}")

    ultimo_orden=order

    if mejora_coste < 1e-3 * mejor_coste_global:
        break

# --- BUCLE DE DIBUJADO Y ANÁLISIS DE TEMPERATURAS ---
def visualizar_espectro_liouvilliano(evals, order):
    plt.figure(figsize=(10, 8))
    plt.scatter(evals.real, evals.imag, c=evals.real, cmap='viridis_r', s=10, alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.colorbar(label='Parte Real (Tasa de Decaimiento)')
    plt.title(f"Espectro del Liouvilliano (Autovalores) - Orden {order}")
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(output_path+f"results/evals/liouvillian_evals_g{order}.png")
    plt.close()

with open(output_path+"results/temps/temps.csv", "w") as f_temps:
    for order in range(2, ultimo_orden + 1):
        print(f" Dibujando espectro y distancias orden {order}...")
        
        C = np.array(qt.qload(output_path+f"results/steadies/steady_g{order}").full())
        fig, ax1 = plt.subplots(figsize=(10,10))
        im1 = ax1.imshow(np.abs(C), cmap='hot')
        fig.colorbar(im1, ax=ax1, orientation='vertical')
        fig.savefig(output_path+f"results/steady_g{order}.png")
        plt.close(fig)

        
        try:
            evals = np.load(output_path+f"results/evals/liouvillian_evals_g{order}.npy")
            visualizar_espectro_liouvilliano(evals, order)
        except Exception as e:
            print(f"Error ploteando evals: {e}")

print("\n¡Simulación completada con éxito!")