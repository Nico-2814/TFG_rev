import numpy as np
from scipy import *
from qutip import *
import h5py
import time

Gamma = 0.1
T = 1.0

def spec(w):
    if w == 0.0:
        return Gamma * T 
        
    n_th = 1.0 / (np.exp(np.abs(w) / T) - 1.0)
    
    if w > 0:
        return Gamma * w * (1.0 + n_th)
    else:
        return Gamma * np.abs(w) * n_th

#Dynamics
from definitions_HCE import timestamps, K, N_steps, N_c, N_Q, max_step, Pi_P, Pi_Q, I_c, I_Q, hbar

H = Qobj(np.load("hamiltonians/hamiltonian_5_2.npy", allow_pickle=True)).to("csr")
rho_0 = Qobj(np.load("HCE_5.npy")).to("csr")
rho_0= rho_0/rho_0.tr()
print(rho_0.tr())
rho_0.dims=H.dims = [[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]]
a= destroy(N_c)
a_d=a.dag()
Q = 1/np.sqrt(2)*(a+a_d)
Q2op1 = tensor(qeye(N_Q), qeye(N_Q), Q*Q, qeye(N_c))
Q2op2 = tensor(qeye(N_Q), qeye(N_Q), qeye(N_c), Q*Q)


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

def distance_to_HCE(t, rho):
    dif = (rho-rho_0)
    return (dif*dif).tr().real

def purity(t, rho):
    return rho.purity()

def entropy_fun(t, rho):
    return entropy_vn(rho)

index=[0]
def saver(t, rho):
    if index[0] % 100 == 0 and t>300:
        np.save(f"rhos/rho{t:.2f}.npy", rho.full())
    index[0] += 1
    return 0


ops = rhos_list+[Q2op1, Q2op2, distance_to_HCE, purity, saver] #he quitado entropía

L_dyn=liouvillian(H)
H2=Qobj(np.load("hamiltonians/hamiltonian_5.npy", allow_pickle=True))
H2.dims = [[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]]
a_ops=[[hbar*tensor(I_Q, I_Q, Pi_Q), spec], [hbar*tensor(I_Q, I_Q, Pi_P), spec], [hbar*tensor(sigmay(), I_Q, I_c, I_c), spec], [hbar*tensor(I_Q, sigmay(), I_c, I_c), spec]]
R, ekets=bloch_redfield_tensor(H2, a_ops)
R=R-liouvillian(H2)
L=L_dyn+R

t1=time.time()

print("Iniciado simulación")

Time_Evolution = mesolve (L, rho_0, timestamps, e_ops = ops, options = {"progress_bar": True, "max_step": max_step, "store_states": False, "store_final_state": True})
results=Time_Evolution.expect
qsave(Time_Evolution.final_state, "final_state_br")

with h5py.File("results/test_BR_2.h5", "w") as f:
    '''
    q1 = f.create_dataset("qubit_1", shape=(N_steps, 3), dtype=np.complex128)
    q2 = f.create_dataset("qubit_2", shape=(N_steps, 3), dtype=np.complex128)
    cq = f.create_dataset("classical_q", shape=(N_steps, 1), dtype=np.complex128)
    cp = f.create_dataset("classical_p", shape=(N_steps, 1), dtype=np.complex128)
    dist = f.create_dataset("distance_to_HCE", shape=(N_steps, 1), dtype=float)
    S = f.create_dataset("entropy", shape=(N_steps, 1), dtype = float)
    P = f.create_dataset("purity", shape = (N_steps, 1), dtype = np.complex128)
    f.create_dataset("tlist", data=timestamps)
    '''
    
    f.create_dataset("rho10", data=np.real(np.asarray(results[0])), dtype=float)
    f.create_dataset("rho20", data=np.real(np.asarray(results[1])), dtype=float)
    f.create_dataset("rho30", data=np.real(np.asarray(results[2])), dtype=float)
    f.create_dataset("rho01", data=np.real(np.asarray(results[3])), dtype=float)
    f.create_dataset("rho02", data=np.real(np.asarray(results[4])), dtype=float)
    f.create_dataset("rho03", data=np.real(np.asarray(results[5])), dtype=float)
    f.create_dataset("rho11", data=np.real(np.asarray(results[6])), dtype=float)
    f.create_dataset("rho12", data=np.real(np.asarray(results[7])), dtype=float)
    f.create_dataset("rho13", data=np.real(np.asarray(results[8])), dtype=float)
    f.create_dataset("rho21", data=np.real(np.asarray(results[9])), dtype=float)
    f.create_dataset("rho22", data=np.real(np.asarray(results[10])), dtype=float)
    f.create_dataset("rho23", data=np.real(np.asarray(results[11])), dtype=float)
    f.create_dataset("rho31", data=np.real(np.asarray(results[12])), dtype=float)
    f.create_dataset("rho32", data=np.real(np.asarray(results[13])), dtype=float)
    f.create_dataset("rho33", data=np.real(np.asarray(results[14])), dtype=float)   
    f.create_dataset("classical_q", data=np.real(np.asarray(results[15])), dtype=float)
    f.create_dataset("classical_p", data=np.real(np.asarray(results[16])), dtype=float)
    f.create_dataset("distance_to_HCE", data=np.real(np.asarray(results[17])), dtype=float)
    # S = f.create_dataset("entropy", data=np.real(np.asarray(results[18])), dtype=float)
    f.create_dataset("purity", data=np.real(np.asarray(results[18])), dtype=float)
    f.create_dataset("tlist", data=np.asarray(timestamps, dtype=float))
    f.create_dataset("hbar", data=float(hbar))
    f.create_dataset("N_c", data=int(N_c))
    f.create_dataset("N_Q", data=int(N_Q))

print(f"Tiempo transcurrido: {time.time()-t1} s.")