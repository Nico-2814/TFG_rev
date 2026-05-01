import numpy as np
import subprocess
import os
import qutip as qt
from time import time

N_c=15
N_Q=2
N_iters=10
error = 0.01 

dim=N_c**2*N_Q**2

out_def = subprocess.run(["python", "definitions_HCE_for_variations.py", "0.1"])
if out_def.returncode != 0:
    print("Error creando definiciones:", out_def.stderr)

for n_iter in range(N_iters):
    rho = qt.Qobj(np.load("HCE.npy")).to("csr")
    rho= (1 - error) * rho + error * qt.rand_dm(dim, density=0.05, seed=int(time()))
    
    os.mkdir(f"variations/var{n_iter}")
    np.save(f"variations/var{n_iter}/rho.npy", np.array(rho.full()))
    
    out = subprocess.run(["python", "main_HCE_for_variations.py", str(n_iter), str(N_c), str(N_Q), "1000", "10000", "0.01", "0.1"])


    if out.returncode != 0:
            print(f"¡El subproceso falló en i={n_iter}!")
            print("Motivo del error:")
            print(out.stderr) 
            