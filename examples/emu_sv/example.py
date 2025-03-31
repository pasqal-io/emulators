from emu_sv.state_vector import StateVector, inner

from emu_sv.dense_operator import DenseOperator

import torch

dtype = torch.complex128

state1 = StateVector(
    torch.tensor(
        [1 / torch.sqrt(torch.tensor(2.0)), 0, 0, 1 / torch.sqrt(torch.tensor(2.0))],
        dtype=dtype,
    )
)

state2 = StateVector(torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=dtype))


print(state1)
print(state2)
print("\nsampling state2:", state2.sample())

print("\ninner product:", inner(state1, state2))

# state from string

string = {"rrr": 1.0, "ggg": 1.0}
basis = ("r", "g")
nqubits = len(list(string.keys())[0])

state = StateVector.from_state_amplitudes(eigenstates=basis, amplitudes=string)
print(state)


print("Operators in Emu-SV")
operations = [
    (
        1.0,
        [
            ({"g" + "r": 2.0, "r" + "g": 2.0}, [0, 2]),  # 2X
            ({"g" + "g": 3.0, "r" + "g": -3.0}, [1]),  # 3Z
        ],
    )
]

basis = {"r", "g"}
N = 3
oper = DenseOperator.from_operator_repr(
    eigenstates=basis,
    n_qudits=N,
    operations=operations,
)
print(oper)
