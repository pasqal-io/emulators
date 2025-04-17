import math
import pytest
import torch
from emu_sv.lindblad_operator import LindbladOperator
from test.utils_testing import dense_rydberg_hamiltonian, nn_interaction_matrix

dtype = torch.complex128
dtype_adp = torch.float64
device = torch.device("cpu")


def test_ham_matmul_density():
    """H @ 𝜌, with out lindblad operators"""
    torch.manual_seed(234)
    nqubits = 10
    omegas = torch.rand(nqubits, dtype=dtype_adp, device=device).to(torch.complex128)
    deltas = torch.rand(nqubits, dtype=dtype_adp, device=device).to(torch.complex128)
    phis = torch.zeros(nqubits, dtype=dtype_adp, device=device).to(torch.complex128)
    pulser_linblad = [torch.zeros(2, 2, dtype=dtype, device=device)]
    interaction_matrix = nn_interaction_matrix(nqubits)

    rho = torch.rand(2**nqubits, 2**nqubits, dtype=dtype, device=device)
    ham_lind = LindbladOperator(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        pulser_linblads=pulser_linblad,
        interaction_matrix=interaction_matrix,
        device=device,
    )
    result = ham_lind @ rho

    ham = dense_rydberg_hamiltonian(
        omegas=omegas, deltas=deltas, phis=phis, interaction_matrix=interaction_matrix
    )
    h_rho = ham @ rho

    assert torch.allclose(result, h_rho)


@pytest.mark.parametrize(
    "target_qubit",
    [0, 1, 2, 3, 4, 5, 6, 7],
)
def test_apply_local_operator_on_target_qubit(target_qubit):
    nqubits = 8  # Total number of qubits
    target = target_qubit  # Qubit on which the lindblad_op acts

    omegas = torch.zeros(nqubits, dtype=dtype_adp, device=device).to(torch.complex128)
    deltas = torch.zeros(nqubits, dtype=dtype_adp, device=device).to(torch.complex128)
    phis = torch.zeros(nqubits, dtype=dtype_adp, device=device).to(torch.complex128)
    pulser_linblad = torch.zeros(2**nqubits, dtype=dtype, device=device)
    interaction_matrix = torch.zeros(nqubits, nqubits)

    ham_lind = LindbladOperator(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        pulser_linblads=pulser_linblad,
        interaction_matrix=interaction_matrix,
        device=device,
    )

    # Define a random 2x2 complex matrix
    lindblad_op = torch.randn(2, 2, dtype=dtype, device=device)
    rho = torch.randn(2**nqubits, 2**nqubits, dtype=dtype, device=device)

    # Apply the local operator
    updated_rho = ham_lind.apply_local_operator_to_density_matrix(
        density_matrix=rho, local_op=lindblad_op, target_qubit=target
    )

    ident = torch.eye(2, dtype=dtype, device=device)

    lista = []
    for i in range(nqubits):
        if i == target:
            lista.append(lindblad_op)
        else:
            lista.append(ident)

    res = torch.kron(lista[0], lista[1])
    for i in range(2, nqubits):
        res = torch.kron(res, lista[i])

    assert torch.allclose(updated_rho, res @ rho)


def test_lindblads():
    """Testing 0.5*i*(∑ₖ Lₖ^† Lₖ) 𝜌 part"""
    torch.manual_seed(234)
    nqubits = 8
    omegas = torch.zeros(nqubits, dtype=dtype_adp, device=device).to(torch.complex128)
    deltas = torch.zeros(nqubits, dtype=dtype_adp, device=device).to(torch.complex128)
    phis = torch.zeros(nqubits, dtype=dtype_adp, device=device).to(torch.complex128)
    pulser_linblads = [
        math.sqrt(1 / 3)
        * torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device),
        math.sqrt(1 / 3)
        * torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device),
        math.sqrt(1 / 3)
        * torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype, device=device),
    ]
    interaction_matrix = torch.zeros(nqubits, nqubits, dtype=dtype, device=device)

    rho = torch.rand(2**nqubits, 2**nqubits, dtype=dtype, device=device)

    ham_lind = LindbladOperator(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        pulser_linblads=pulser_linblads,
        interaction_matrix=interaction_matrix,
        device=device,
    )
    result = ham_lind @ rho

    # 0.5*i*(∑ₖ Lₖ^† Lₖ) 𝜌 by hand

    ident = torch.eye(2, dtype=dtype, device=device)

    lista = []
    for lind in pulser_linblads:
        for i in range(nqubits):
            identities = [ident] * nqubits
            identities[i] = lind.conj().T.contiguous() @ lind
            lista.append(identities)

    result_kron = torch.zeros_like(rho)
    for num, _ in enumerate(lista):
        res = torch.kron(lista[num][0], lista[num][1])
        for i in range(2, nqubits):
            res = torch.kron(res, lista[num][i])
        result_kron += 0.5j * res @ rho

    assert torch.allclose(result, result_kron)
