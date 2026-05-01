from qutip import *
import numpy as np
from scipy.linalg import sinm, cosm
import sys

#Dimensions
N_c=10 #Number of classical states for each oscillator
S=0.5 #Spin
N_Q=int(2*S+1)

#Operators
I_c = qeye(N_c)
I_Q = qeye(N_Q)

a=destroy(N_c)
a_q = tensor(a, I_c)
a_d_q = a_q.dag()
a_p = tensor(I_c, a)
a_d_p = a_p.dag()

Q = 1/np.sqrt(2)*(a_q+a_d_q)
P = 1/np.sqrt(2)*(a_p+a_d_p)
Pi_Q = -1j/np.sqrt(2)*(a_q-a_d_q)
Pi_P = -1j/np.sqrt(2)*(a_p-a_d_p)

#Kraus Operators
hbar=float(sys.argv[1])
J=1
h=1
K1=tensor(hbar*(J*tensor(sigmaz(), sigmaz())+h*(tensor(sigmax(), I_Q)+tensor(I_Q, sigmax()))), I_c, I_c)
K2=tensor(I_Q, I_Q, Pi_P+Pi_Q)

qsave(K1, "variations/K1")
qsave(K2, "variations/K2")
