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
'''
a = np.zeros((N_c, N_c))
for i in range(N_c-1):
    a[i][i+1]=np.sqrt(i+1)
    '''

a=destroy(N_c)
a_q = tensor(a, I_c)
a_d_q = a_q.dag()
a_p = tensor(I_c, a)
a_d_p = a_p.dag()

Q = 1/np.sqrt(2)*(a_q+a_d_q)
P = 1/np.sqrt(2)*(a_p+a_d_p)
Pi_Q = -1j/np.sqrt(2)*(a_q-a_d_q)
Pi_P = -1j/np.sqrt(2)*(a_p-a_d_p)
'''
cQ = Q.cosm()
sQ = Q.sinm()
Id_dim = qeye(N_c*N_c)
Id_dim.dims = [[N_c, N_c], [N_c, N_c]]
E1 = np.linalg.inv(Id_dim+Q*Q)
E2 = E1 + Id_dim+ 0.1*Q*Q
P1 = qeye(N_Q*N_c*N_c)+tensor(sigmaz(), cQ)+tensor(sigmax(), sQ)
P2 = qeye(N_Q*N_c*N_c)-tensor(sigmaz(), cQ)-tensor(sigmax(), sQ)
'''

#Quantum Observables
S2_i=S*(S+1)*qeye(N_Q)
'''
S_z_i = np.zeros((N_Q, N_Q))
for i in range (N_Q):
    S_z_i[i][i]=S-i
    '''

S_z_i = jmat(S, 'z')
S_p_i = jmat(S, '+')
S_m_i = jmat(S, '-')


S2 = tensor(S2_i, I_Q)+tensor(I_Q, S2_i)+tensor(S_p_i, S_m_i)+tensor(S_m_i, S_p_i)+tensor(S_z_i, S_z_i)


#Hamiltonian
Id = tensor (I_c, I_c)
H_c = P*Pi_Q-Q*Pi_P
H_Q = tensor(sigmax(), sigmax())
#H_CQ = 0.5*E1*(tensor(P1, qeye(N_Q))+ tensor(qeye(N_Q), P1))+0.5*E2*(tensor(P2, qeye(N_Q)+tensor(qeye(2), P2)))

#H = tensor(H_c, I_Q, I_Q)+H_CQ

#Kraus Operators
hbar=0.1
J=1
h=1
K=tensor(hbar*(J*tensor(sigmaz(), sigmaz())+h*(tensor(sigmax(), I_Q)+tensor(I_Q, sigmax()))), I_c, I_c)

#Initial States

'''
phi_c = tensor(basis(N_c, 1), basis(N_c, 0))
phi_Q = tensor(basis(N_Q, 0), basis(N_Q, 1))

phi_0 = tensor(phi_c, phi_Q).unit()
rho_0 = phi_0 * phi_0.dag()
'''

#rho_0=Qobj(np.loadtxt("HCE.txt", dtype=np.complex128, delimiter="\t"))

#Timeline
T_max = 500
N_steps = 5000
max_step = 0.01

timestamps = np.linspace (0, T_max, N_steps)
