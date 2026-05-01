from qutip import *
import numpy as np
from scipy.linalg import sinm, cosm

#Dimensions
N_c=5 #Number of classical states for each oscillator
S=0.5 #Spin
N_Q=int(2*S+1)

#Operators
I_c = qeye(N_c)
I_Q = qeye(N_Q)

#Classical Observables
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
hbar=1
J=1
h=1
K1=tensor(hbar*(J*tensor(sigmaz(), sigmaz())+h*(tensor(sigmax(), I_Q)+tensor(I_Q, sigmax()))), I_c, I_c)


#Times
T_max = 500
N_steps = 5000 #number of measures
max_step = 0.1

timestamps = np.linspace (0, T_max, N_steps)