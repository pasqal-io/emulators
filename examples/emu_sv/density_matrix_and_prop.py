from emu_sv import DensityMatrix, StateVector

import torch

dtype = torch.complex128

# small example to show how to create a density matrix
# using emu-sv package

# different ways to create a density matrix
# 1. from a emu-sv state vector
# 2. using the amplitudes and the basis
# 3. creating a ground state density matrix

# 1.- using the state vector
state1 = DensityMatrix.from_state_vector(
    StateVector(
        torch.tensor(
            [1 / torch.sqrt(torch.tensor(2.0)), 0, 0, 1 / torch.sqrt(torch.tensor(2.0))],
            dtype=dtype,
        )
    )
)

state2 = DensityMatrix.from_state_vector(
    StateVector(torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=dtype))
)


print(state1.matrix)
print(state2.matrix)
print("\nsampling state2:", state2.sample())

print("\ninner product:", state1.overlap(state2))


# 2.- using amplitudes and the basis

n_atoms = 2
amplitudes = {"gg": 1.0, "rr": 1.0j}
eigenstates = ("r", "g")
density, input_amplitudes = DensityMatrix._from_state_amplitudes(
    eigenstates=eigenstates, n_qudits=n_atoms, amplitudes=amplitudes
)
print("\nDensity matrix from state amplitudes:\n", density.matrix)
print("\nAmplitudes:\n", input_amplitudes)


# creating a ground state density matrix

n_qubits = 4
ground_state = DensityMatrix.make(n_atoms, gpu=False)
print("\nGround state density matrix:\n", ground_state.matrix)
