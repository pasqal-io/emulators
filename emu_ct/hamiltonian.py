import torch
from emu_ct.mpo import MPO
from emu_ct.qubit_position import dist2
from emu_ct.pulser_adapter import get_qubit_positions
import pulser


"""
Takes a single qubit operator, and creates the first factor in a Hamiltonian
consisting of this single qubit operator and the Rydberg interaction terms.
gate should be on the target device

This returns the shape (1,2,2,3) [Gate, 1, P] where P projects on |1>
"""


def first_factor(gate: torch.Tensor) -> torch.Tensor:
    fac = torch.zeros(1, 2, 2, 3, dtype=gate.dtype)
    fac[0, 0, 0, 1] = 1
    fac[0, 1, 1, 1] = 1
    fac[0, 1, 1, 2] = 1
    fac = fac.to(gate.device)
    fac[0, :, :, 0] = gate
    return fac


"""
Takes a single qubit operator, and creates the last factor in a Hamiltonian
consisting of this single qubit operator and the Rydberg interaction terms.
gate should be on the target device
This returns the shape (3,2,2,1) [1, Gate, P]^T where P projects on |1>
"""


def last_factor(gate: torch.Tensor, scale: float | complex) -> torch.Tensor:
    fac = torch.zeros(3, 2, 2, 1, dtype=gate.dtype)
    fac[0, 0, 0, 0] = 1
    fac[0, 1, 1, 0] = 1
    fac[2, 1, 1, 0] = scale
    fac = fac.to(gate.device)
    fac[1, :, :, 0] = gate
    return fac


"""
Create the Hamiltonian factors in the left half of the MPS that are
not the first factor.
"""


def left_factor(gate: torch.Tensor, scales: list[float]) -> torch.Tensor:
    index = len(scales)
    fac = torch.zeros(index + 2, 2, 2, index + 3, dtype=gate.dtype)
    for i, val in enumerate(scales):
        fac[i + 2, 1, 1, 0] = val  # rydberg interaction with previous qubits
    fac[1, 1, 1, index + 2] = 1  # rydberg interaction with next qubits
    for i in range(index + 2):
        fac[i, 0, 0, i] = 1
        fac[i, 1, 1, i] = 1  # identity matrix to carry the gates of other qubits
    fac = fac.to(gate.device)
    fac[1, :, :, 0] = gate
    return fac


"""
Create the Hamiltonian factors in the right half of the MPS that are
not the last factor.
"""


def right_factor(gate: torch.Tensor, scales: list[float]) -> torch.Tensor:
    index = len(scales)
    fac = torch.zeros(index + 3, 2, 2, index + 2, dtype=gate.dtype)
    for i, val in enumerate(scales):
        fac[1, 1, 1, i + 2] = val  # rydberg interaction with previous qubits
    fac[2, 1, 1, 0] = 1  # rydberg interaction with next qubits
    for i in range(2, index + 2):
        fac[i + 1, 0, 0, i] = 1
        fac[
            i + 1, 1, 1, i
        ] = 1  # identity matrix to carry previous interactions to the next qubits
    fac[0, 0, 0, 0] = 1
    fac[0, 1, 1, 0] = 1  # identity to carry the next gates to the previous qubits
    fac[1, 0, 0, 1] = 1
    fac[1, 1, 1, 1] = 1  # identity to carry previous gates to next qubits
    fac = fac.to(gate.device)
    fac[1, :, :, 0] = gate
    return fac


"""
Create the Hamiltonian factor at index (floor((nqubits)/2)) of the MPS for nqubits > 2
len(scales_l)=m, len(scales_r)=n, size(scales_mat)=(m,n)
"""


def middle_factor(
    gate: torch.Tensor,
    scales_l: list[float],
    scales_r: list[float],
    scales_mat: list[list[float]],
) -> torch.Tensor:
    fac = torch.zeros(len(scales_l) + 2, 2, 2, len(scales_r) + 2, dtype=gate.dtype)
    for i, val in enumerate(scales_r):
        fac[1, 1, 1, i + 2] = val  # rydberg interaction with previous qubits
    for i, val in enumerate(scales_l):
        fac[i + 2, 1, 1, 0] = val  # rydberg interaction with next qubits
    for i, row in enumerate(scales_mat):
        for j, val in enumerate(row):
            fac[
                i + 2, 0, 0, j + 2
            ] = val  # rydberg interaction of previous with next qubits
            fac[i + 2, 1, 1, j + 2] = val
    fac[0, 0, 0, 0] = 1
    fac[0, 1, 1, 0] = 1  # identity to carry the next gates to the previous qubits
    fac[1, 0, 0, 1] = 1
    fac[1, 1, 1, 1] = 1  # identity to carry previous gates to next qubits
    fac = fac.to(gate.device)
    fac[1, :, :, 0] = gate
    return fac


def rydberg_interaction(sequence: pulser.Sequence) -> torch.Tensor:
    num_qubits = len(sequence.register.qubit_ids)

    c6 = sequence.device.interaction_coeff

    qubit_positions = get_qubit_positions(sequence.register)
    interaction_matrix = torch.zeros(num_qubits, num_qubits)

    for numi in range(len(qubit_positions)):
        for numj in range(numi + 1, len(qubit_positions)):
            interaction_matrix[numi][numj] = (
                c6 / dist2(qubit_positions[numi], qubit_positions[numj]) ** 3
            )

    return interaction_matrix


"""
Returns an MPO representing the Hamiltonian specified by omega and delta.

`noise` should be `-1j / 2. * sum(lind.T.conj() @ lind for lind in lindbladians)`.
"""


def make_H(
    interaction_matrix: torch.tensor,
    omega: torch.Tensor,
    delta: torch.Tensor,
    num_devices_to_use: int,
    /,
    noise: torch.Tensor = torch.zeros(2, 2),
) -> MPO:
    assert noise.shape == (2, 2)

    nqubits = interaction_matrix.size(dim=1)
    dtype = omega[0].dtype
    device = omega[0].device

    sx = torch.tensor([[0, 0.5], [0.5, 0]], dtype=dtype, device=device)
    pu = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype, device=device)
    cores = [first_factor(omega[0] * sx - delta[0] * pu + noise)]

    if nqubits > 2:
        for i in range(1, nqubits // 2):

            cores.append(
                left_factor(
                    omega[i] * sx - delta[i] * pu + noise,
                    [interaction_matrix[j, i] for j in range(i)],
                )
            )

        i = nqubits // 2
        cores.append(
            middle_factor(
                omega[i] * sx - delta[i] * pu + noise,
                [interaction_matrix[j, i] for j in range(i)],
                [interaction_matrix[i, j] for j in range(i + 1, nqubits)],
                [
                    [interaction_matrix[k, j] for j in range(i + 1, nqubits)]
                    for k in range(i)
                ],
            )
        )

        for i in range(nqubits // 2 + 1, nqubits - 1):
            cores.append(
                right_factor(
                    omega[i] * sx - delta[i] * pu + noise,
                    [interaction_matrix[i, j] for j in range(i + 1, nqubits)],
                )
            )

    scale = 1.0
    if nqubits == 2:
        scale = interaction_matrix[0, 1]
    cores.append(last_factor(omega[-1] * sx - delta[-1] * pu + noise, scale))
    return MPO(cores)
