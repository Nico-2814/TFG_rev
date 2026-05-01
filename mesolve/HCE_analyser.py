import numpy as np
import qutip as qt
import h5py
import matplotlib.pyplot as plt

with h5py.File("results/h_1_br_2.h5", "r") as f:
    def dist_to_HCE():
        tlist = f["tlist"][:]
        hbar=f["hbar"][()]
        dist_t = f["distance_to_HCE"][:]
        N_c=f["N_c"][()]
        N_Q=f["N_Q"][()]

        dim=(N_c*N_Q)**2

        fig, ax = plt.subplots()
        ax.plot(tlist*hbar, dist_t/dim)
        ax.set_xlabel('t')
        ax.set_ylabel(r'$||\rho(t)-\rho(0)||_2^2$')
        ax.set_title('Evolución de la distancia al HCE')

        plt.savefig("dist.png")

    def Q_ev():
        tlist = f["tlist"][:]
        hbar=f["hbar"][()]
        Q_mat = f["classical_q"][:]

        fig, ax = plt.subplots()
        ax.plot(tlist*hbar, Q_mat)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\langle Q^2\rangle$")
        ax.set_title(r"Evolución de $\langle Q^2\rangle$")

        plt.savefig("Q2.png")

    def dist_to_steady():#Solo en algunas simulaciones
        tlist = f["tlist"][:]
        hbar=f["hbar"][()]
        dist_t = f["distance_to_steady_state"][:]

        fig, ax = plt.subplots()
        ax.plot(tlist*hbar, dist_t)
        ax.set_xlabel('t')
        ax.set_ylabel(r'$||\rho(t)-\rho^*||_2^2$')
        ax.set_title('Evolución de la distancia al estado estable')

        plt.savefig("steady.png")


    def P_ev():
        from definitions_HCE import N_steps

        tlist = f["tlist"][:]
        hbar=f["hbar"][()]
        P_mat = f["classical_p"][:]

        fig, ax = plt.subplots()
        ax.plot(tlist*hbar, P_mat)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\langle P^2\rangle$")
        ax.set_title(r"Evolución de $\langle P^2\rangle$")

        plt.savefig("P2.png")
    
    def qubits_comps_pair():
        hbar=f["hbar"][()]
        tlist = f["tlist"][:]*hbar

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5,3, figsize=(15,25))
        ax1.plot(tlist, f["rho10"][:])
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel(r"$\rho_{10}$")
        ax1.set_title(r"Evolución de $\rho_{10}$")
        

        ax2.plot(tlist, f["rho20"][:])
        ax2.set_xlabel(r"$t$")
        ax2.set_ylabel(r"$\rho_{20}$")
        ax2.set_title(r"Evolución de $\rho_{20}$")

        ax3.plot(tlist, f["rho30"][:])
        ax3.set_xlabel(r"$t$")
        ax3.set_ylabel(r"$\rho_{30}$")
        ax3.set_title(r"Evolución de $\rho_{30}$")

        ax4.plot(tlist, f["rho01"][:])
        ax4.set_xlabel(r"$t$")
        ax4.set_ylabel(r"$\rho_{01}$")
        ax4.set_title(r"Evolución de $\rho_{01}$")

        ax5.plot(tlist, f["rho02"][:])
        ax5.set_xlabel(r"$t$")
        ax5.set_ylabel(r"$\rho_{02}$")
        ax5.set_title(r"Evolución de $\rho_{02}$")

        ax6.plot(tlist, f["rho03"][:])
        ax6.set_xlabel(r"$t$")
        ax6.set_ylabel(r"$\rho_{03}$")
        ax6.set_title(r"Evolución de $\rho_{03}$")

        ax7.plot(tlist, f["rho11"][:])
        ax7.set_xlabel(r"$t$")
        ax7.set_ylabel(r"$\rho_{11}$")
        ax7.set_title(r"Evolución de $\rho_{11}$")

        ax8.plot(tlist, f["rho12"][:])
        ax8.set_xlabel(r"$t$")
        ax8.set_ylabel(r"$\rho_{12}$")
        ax8.set_title(r"Evolución de $\rho_{12}$")

        ax9.plot(tlist, f["rho13"][:])
        ax9.set_xlabel(r"$t$")
        ax9.set_ylabel(r"$\rho_{13}$")
        ax9.set_title(r"Evolución de $\rho_{13}$")

        ax10.plot(tlist, f["rho21"][:])
        ax10.set_xlabel(r"$t$")
        ax10.set_ylabel(r"$\rho_{21}$")
        ax10.set_title(r"Evolución de $\rho_{21}$")

        ax11.plot(tlist, f["rho22"][:])
        ax11.set_xlabel(r"$t$")
        ax11.set_ylabel(r"$\rho_{22}$")
        ax11.set_title(r"Evolución de $\rho_{22}$")

        ax12.plot(tlist, f["rho23"][:])
        ax12.set_xlabel(r"$t$")
        ax12.set_ylabel(r"$\rho_{23}$")
        ax12.set_title(r"Evolución de $\rho_{23}$")

        ax13.plot(tlist, f["rho31"][:])
        ax13.set_xlabel(r"$t$")
        ax13.set_ylabel(r"$\rho_{31}$")
        ax13.set_title(r"Evolución de $\rho_{31}$")

        ax14.plot(tlist, f["rho32"][:])
        ax14.set_xlabel(r"$t$")
        ax14.set_ylabel(r"$\rho_{32}")
        ax14.set_title(r"Evolución de $\rho_{32}$")

        ax15.plot(tlist, f["rho33"][:])
        ax15.set_xlabel(r"$t$")
        ax15.set_ylabel(r"$\rho_{33}$")
        ax15.set_title(r"Evolución de $\rho_{33}$")

        fig.tight_layout()

        plt.savefig("rhos_pair.png")
    
    def entropy():
        S = f["entropy"][:]
        tlist = f["tlist"][:]
        hbar=f["hbar"][()]

        fig, ax = plt.subplots()
        ax.plot(tlist*hbar, S)
        ax.set_xlabel('t')
        ax.set_ylabel(r'$S$')
        ax.set_title('Evolución de la entropía')

        plt.savefig("S.png")

    def purity():
        P = f["purity"][:]
        tlist = f["tlist"][:]
        hbar=f["hbar"][()]

        fig, ax = plt.subplots()
        ax.plot(tlist*hbar, P)
        ax.set_xlabel('t')
        ax.set_ylabel(r'Tr$\rho^2$')
        ax.set_title('Evolución de la pureza')

        plt.savefig("purity.png")

    dist_to_HCE()
    #dist_to_steady()
    Q_ev()
    P_ev()
    qubits_comps_pair()
    entropy()
    purity()