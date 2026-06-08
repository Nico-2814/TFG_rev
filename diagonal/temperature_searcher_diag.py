import subprocess
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

with open("distances_full.csv", "w") as f:
    f.write("beta,dist\n")
    def norm(beta):
        rho=qt.qload("HCEs/HCE_"+str(beta)+"_5_diag")
        rho.dims=[[2,2,5,5],[2,2,5,5]]
        dif = (rho-HCE)
        del rho
        f.write( str(beta)+","+str((dif*dif).tr().real/100)+"\n")

    #subprocess.run(["python", "main.py"])
    #subprocess.run(["python", "opt_HCE_calculator_diag.py"])

    HCE=qt.qload("results/stable_state_2")
    HCE.dims=[[2,2,5,5],[2,2,5,5]]

    betas=np.logspace(-2,2,50)
    for beta in betas:
        norm(beta)

with open("distances_full.csv", "r") as f:
    data=np.loadtxt(f, delimiter=',', skiprows=1)
    bet=data[:,0]
    d=data[:,1]

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(bet, d, 'o')
    ax.set_title(r"distancia al estacionario en función de $\beta$")
    ax.set_xlabel(r"$\beta$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.savefig("HCEs/dists_log_diag_1.png")

