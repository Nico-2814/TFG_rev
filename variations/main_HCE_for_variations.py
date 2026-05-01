import numpy as np
from scipy import *
from qutip import *
import h5py
import time
import sys


H = Qobj(np.loadtxt("hamiltonian.txt")).to("csr")
N_iter=int(sys.argv[1])
rho_0 = Qobj(np.load(f"variations/var{N_iter}/rho.npy", allow_pickle=True))
steady=qload("rhos/mean")
N_c=int(sys.argv[2])
N_Q=int(sys.argv[3])

#Dynamics
K1=qload("variations/K1")
K2=qload("variations/K2")
        
timestamps=np.linspace(0, float(sys.argv[4]), int(sys.argv[5]))
max_step=float(sys.argv[6])
hbar=float(sys.argv[7])


rho_0= rho_0/rho_0.tr()
rho_0.dims=H.dims=K1.dims=K2.dims=steady.dims = [[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]]
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

def distance_to_steady_state(t, rho):
    dif = (rho-steady)
    return (dif*dif).tr().real

def purity(t, rho):
    return rho.purity()

def entropy_fun(t, rho):
    return entropy_vn(rho)



index=[0]

def saver(t, rho):
    if index[0] % 1000 == 0:
        np.save(f"variations/var{N_iter}/rho{t:.2f}.npy", rho.full())
    index[0] += 1
    return 0

ops = rhos_list+[Q2op1, Q2op2, distance_to_HCE, distance_to_steady_state, entropy_fun, purity, saver] 

t1=time.time()

print("Iniciado simulación")

Time_Evolution = mesolve (H, rho_0, timestamps, c_ops=[K1, K2], e_ops = ops, options = {"progress_bar": True, "max_step": max_step, "store_states": False})
results=Time_Evolution.expect

with h5py.File(f"variations/var{N_iter}/out.h5", "w") as f:
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
    f.create_dataset("distance_to_steady_state", data=np.real(np.asarray(results[18])), dtype=float)
    f.create_dataset("entropy", data=np.real(np.asarray(results[18])), dtype=float)
    f.create_dataset("purity", data=np.real(np.asarray(results[19])), dtype=float)
    f.create_dataset("tlist", data=np.asarray(timestamps, dtype=float))
    f.create_dataset("hbar", data=float(hbar))
    f.create_dataset("N_c", data=int(N_c))
    f.create_dataset("N_Q", data=int(N_Q))

print(f"Tiempo transcurrido para iteración {N_iter}: {time.time()-t1} s.")