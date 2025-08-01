

#%%


import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
# from qutip.qip.operations import rz, cz_gate
# import cmath

# define fluxonium A
qbta = scq.Fluxonium(
    EC=1.06,
    EJ=4.62,
    EL=1.09,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=4,
)

# define fluxonium B
qbtb = scq.Fluxonium(
    EC=1.03,
    EJ=5.05,
    EL=1.88,
    flux=0.5,
    cutoff=110,
    truncated_dim=4,
)

# define the common Hilbert space
hilbertspace = scq.HilbertSpace([qbta, qbtb])


# add interaction between two qubits
hilbertspace.add_interaction(
    g_strength=0.28,    op1=qbta.n_operator,
    op2=qbtb.n_operator,
)

# generate spectrum lookup table
hilbertspace.generate_lookup()

# Hamiltonian in dressed eigenbasis
(evals,) = hilbertspace["evals"]
# The factor of 2pi converts the energy to GHz so that the time is in units of ns
diag_dressed_hamiltonian = (
        2 * np.pi * qt.Qobj(np.diag(evals),
        dims=[hilbertspace.subsystem_dims] * 2)
)

# The matrix representations can be truncated further for the simulation
total_truncation =15

# truncate operators to desired dimension
def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    return qt.Qobj(operator[:dimension, :dimension])

diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)





# get the representation of the n_a operator in the dressed eigenbasis of the composite system
n_a = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbta.n_operator)
n_b = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbtb.n_operator)

# truncate the operator after expressing in the dressed basis to speed up the simulation
n_a = truncate(n_a, total_truncation)
n_b = truncate(n_b, total_truncation)

# n_a_bare=hilbertspace.op_in_bare_eigenbasis(op_callable_or_tuple=qbta.n_operator)
# product_states_unsorted = [(0, 0), (1, 0), (0, 1),(2,0), (1, 1),(0,3) , (2,1),(0,2),(3,0),(4,0),(1,2),(3,1),(2,2),(4,1),(3,2),(0,4),(1,4),(2,3),(1,3)]
product_states_unsorted = [(0, 0), (1, 0), (0, 1),(2,0), (1, 1),(2,1),(0,2),(1,2), (0,3),(3,0)]

idxs_unsorted = [hilbertspace.dressed_index((s1, s2)) for (s1, s2) in product_states_unsorted]

paired_data = list(zip(idxs_unsorted, product_states_unsorted))
sorted_data = sorted(paired_data, key=lambda x: x[0])
product_states = [data[1] for data in sorted_data]
idxs = [data[0] for data in sorted_data]
#sort after writing, paired data sort

states = [qt.basis(total_truncation, idx) for idx in idxs]

index_to_state = {idx: f'{state[0]}{state[1]}' for idx, state in zip(idxs, product_states)}
# Function to get idsx value from (i, j) tuple
def get_idx(state_tuple):
    state_string = f'{state_tuple[0]}{state_tuple[1]}'
    for idx, state_str in index_to_state.items():
        if state_str == state_string:
            return idx
    return None  # Return None if state_tuple is not found


dim=total_truncation
Omega =np.zeros((dim ,dim))
freq_tran = np.zeros((dim ,dim))
computational_subspace = states[:5] 
def transition_frequency(s0: int, s1: int) -> float:
    return (
        (
            hilbertspace.energy_by_dressed_index(s1)
            - hilbertspace.energy_by_dressed_index(s0)
        )
        * 2
        * np.pi
    )
# Nested loop for i and j
for i in range(dim):
    for j in range(i+1, dim):
        # Calculate transition energy w for each pair i, j
        w = transition_frequency(i, j)/6.28 
        Omega[i][j] = w



Delta1 = 1000*(Omega[get_idx((1,0)),get_idx((2,0))]-Omega[get_idx((1,1)),get_idx((2,1))])
Delta2 = 1000*(Omega[get_idx((0,1)),get_idx((0,2))]-Omega[get_idx((1,1)),get_idx((1,2))])
# Delta3 = 1000*(Omega[get_idx((0,0)),get_idx((0,3))]-Omega[get_idx((1,0)),get_idx((1,3))])

Static_ZZ = 1000*(Omega[get_idx((1,0)),get_idx((1,1))]-Omega[get_idx((0,0)),get_idx((0,1))]) #MHz

#%% Fundamental speed limit calculation

n_diff = abs(np.round(n_a[0,2],5)-np.round(n_a[1,4],5))
n_A_01 = np.round(qbta.n_operator(energy_esys=True)[0][1],5)

bare_states_a = qbta.eigenvals()-qbta.eigenvals()[0]
bare_states_b = qbtb.eigenvals()-qbtb.eigenvals()[0]
detuning  = abs(bare_states_a[1]-bare_states_b[1])

t_fsl = abs(n_A_01*2*np.pi/n_diff/detuning)



#%% FT 

import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
# from qutip.qip.operations import rz, cz_gate
# import cmath

# define fluxonium A
qbta = scq.Fluxonium(
    EC=1,
    EJ=5.8416,
    EL=1.1127,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)
qbtb = scq.Transmon(
     EJ=16.69,
     EC=0.2,
     ng=0,
     ncut=110,
     truncated_dim=20)


#Calculating Matrix element

bare_states_a = qbta.eigenvals()-qbta.eigenvals()[0]
w_A_01 = bare_states_a[1]-bare_states_a[0]
w_A_03 = bare_states_a[3]-bare_states_a[0]
w_A_12 = bare_states_a[2]-bare_states_a[1]
Jc = 1#0.025
n_A_01 = abs(np.round(qbta.n_operator(energy_esys=True)[0][1],5))
n_A_03 = abs(np.round(qbta.n_operator(energy_esys=True)[0][3],5))
n_A_12 = abs(np.round(qbta.n_operator(energy_esys=True)[1][2],5))

bare_states_b = qbtb.eigenvals()-qbtb.eigenvals()[0]
w_B_01 = bare_states_b[1]-bare_states_b[0]
w_B_03 = bare_states_b[3]-bare_states_b[0]
w_B_12 = bare_states_b[2]-bare_states_b[1]

n_B_01 = abs(np.round(qbtb.n_operator(energy_esys=True)[0][1],5))
n_B_03 = abs(np.round(qbtb.n_operator(energy_esys=True)[0][3],5))
n_B_12 = abs(np.round(qbtb.n_operator(energy_esys=True)[1][2],5))



w_B_values = np.linspace(0, 8, 101)

# Initialize empty lists to store the results
results = []
results2 = []

# Calculate the expressions for each w_B value
for w_B in w_B_values:
    result = -Jc*((n_A_01**2 * w_A_01) / (w_A_01**2 - w_B**2) + (n_A_03**2 * w_A_03) / (w_A_03**2 - w_B**2))
    results.append(result)

    result2 =Jc*((n_A_01**2 * w_A_01) / (w_A_01**2 - w_B**2) - (n_A_12**2 * w_A_12) / (w_A_12**2 - w_B**2))
    results2.append(result2)

# Convert the results and w_B values to NumPy arrays for plotting
results = np.array(results)
results2 = np.array(results2)
w_B_values = np.array(w_B_values)








# Plot result vs. w_B and result2 vs. w_B on the same graph
plt.figure()
plt.plot(w_B_values, results,linewidth = 2, label='<00|n_A|01>/const')
plt.plot(w_B_values, results2,linewidth = 2, label='<10|n_A|11>/const')
plt.xlabel('w_B')
plt.ylabel('Results')
plt.ylim([-2, 2])
plt.title('Results vs. w_B')
plt.legend()
plt.grid(False)
plt.show()

#%%

qbta = scq.Fluxonium(
    EC=1,
    EJ=4.43,
    EL=.8,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)
qbtb = scq.Transmon(
     EJ=13,
     EC=0.2,
     ng=0,
     ncut=110,
     truncated_dim=20)

hbar = 1


#Calculating Matrix element

bare_states_a = 6.28*(qbta.eigenvals()-qbta.eigenvals()[0])
w_A_01 = 1e9*(bare_states_a[1]-bare_states_a[0])
w_A_03 = 1e9*(bare_states_a[3]-bare_states_a[0])
w_A_12 = 1e9*(bare_states_a[2]-bare_states_a[1])
Jc = 0.025e6
n_A_01 = (np.round(qbta.n_operator(energy_esys=True)[0][1],5))
n_A_03 = (np.round(qbta.n_operator(energy_esys=True)[0][3],5))
n_A_12 = (np.round(qbta.n_operator(energy_esys=True)[1][2],5))

bare_states_b = 6.28*(qbtb.eigenvals()-qbtb.eigenvals()[0])
w_B_01 =1e9*(bare_states_b[1]-bare_states_b[0])
w_B_03 = 1e9*(bare_states_b[3]-bare_states_b[0])
w_B_12 = 1e9*(bare_states_b[2]-bare_states_b[1])

n_B_01 = (np.round(qbtb.n_operator(energy_esys=True)[0][1],5))
n_B_03 = (np.round(qbtb.n_operator(energy_esys=True)[0][3],5))
n_B_12 = (np.round(qbtb.n_operator(energy_esys=True)[1][2],5))


nA_00_01 = Jc*-2*1j/hbar*n_B_01*((n_A_01**2*w_A_01)/(w_A_01**2-w_B_01**2) + (n_A_03**2*w_A_03)/(w_A_03**2-w_B_01**2))
nA_10_11 = Jc*2*1j/hbar*n_B_01*((n_A_01**2*w_A_01)/(w_A_01**2-w_B_01**2) - (n_A_12**2*w_A_12)/(w_A_12**2-w_B_01**2))
nB_00_10 = Jc*2*1j/hbar*n_A_01*((n_B_01**2*w_B_01)/(w_A_01**2-w_B_01**2) + (n_B_03**2*w_B_03)/(w_A_01**2-w_B_03**2))
nB_01_11 = Jc*-2*1j/hbar*n_A_01*((n_B_01**2*w_B_01)/(w_A_01**2-w_B_01**2) + (n_B_12**2*w_B_12)/(w_A_01**2-w_B_12**2))


f = w_B_01

Omega_10_11 = 2*f*abs(nA_10_11 - nA_00_01)
Omega_01_11 = 2*f*abs(nB_01_11 - nB_00_10)

#%% Matrix element ratio for Fluxonium kappa calc

qbta = scq.Fluxonium(
    EC=1,
    EJ=4.43,
    EL=.8,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)
qbtb = scq.Transmon(
     EJ=1.56,
     EC=1,
     ng=0,
     ncut=110,
     truncated_dim=20)