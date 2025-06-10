import math
import pytest
import torch
from emu_sv.lindblad_operator import RydbergLindbladian
from emu_sv import DensityMatrix
from test.utils_testing import (
    dense_rydberg_hamiltonian,
    nn_interaction_matrix,
    list_to_kron,
)


dtype = torch.complex128
dtype_adp = torch.float64
device = "cpu"
gpu = False if device == "cpu" else True


def test_ham_matmul_density():
    """H @ 𝜌, with out lindblad operators"""
    torch.manual_seed(234)
    nqubits = 10
    omegas = torch.rand(nqubits, dtype=dtype_adp, device=device).to(dtype)
    deltas = torch.rand(nqubits, dtype=dtype_adp, device=device).to(dtype)
    phis = torch.rand(nqubits, dtype=dtype_adp, device=device).to(dtype)
    pulser_linblad = [torch.zeros(2, 2, dtype=dtype, device="cpu")]  # always on cpu
    interaction_matrix = nn_interaction_matrix(nqubits)

    rho = torch.rand(2**nqubits, 2**nqubits, dtype=dtype, device=device)
    ham_lind = RydbergLindbladian(
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
    h_rho_left = ham @ rho
    h_rho = h_rho_left - h_rho_left.conj().T
    assert torch.allclose(result, h_rho)


test_atoms = 10


@pytest.mark.parametrize("target_qubit", range(test_atoms))
def test_apply_local_operator_on_target_qubit(target_qubit):
    """Testing the application of a local operator on a target qubit"""
    torch.manual_seed(234)
    nqubits = test_atoms

    omegas = torch.zeros(nqubits, dtype=dtype_adp, device=device).to(dtype)
    deltas = torch.zeros(nqubits, dtype=dtype_adp, device=device).to(dtype)
    phis = torch.zeros(nqubits, dtype=dtype_adp, device=device).to(dtype)
    pulser_linblad = torch.zeros(2**nqubits, dtype=dtype, device="cpu")  # always on cpu
    interaction_matrix = torch.zeros(nqubits, nqubits)

    ham_lind = RydbergLindbladian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        pulser_linblads=pulser_linblad,
        interaction_matrix=interaction_matrix,
        device=device,
    )

    # define a random 2x2 linblad operator
    lindblad_op = torch.randn(2, 2, dtype=dtype, device=device)
    rho = torch.randn(2**nqubits, 2**nqubits, dtype=dtype, device=device)

    # apply the local operator
    updated_rho = ham_lind.apply_local_op_to_density_matrix(
        density_matrix=rho, local_op=lindblad_op, target_qubit=target_qubit
    )

    ident = torch.eye(2, dtype=dtype, device=device)

    lista = []
    for i in range(nqubits):
        if i == target_qubit:
            lista.append(lindblad_op)
        else:
            lista.append(ident)

    res = list_to_kron(lista)

    assert torch.allclose(updated_rho, res @ rho)

    # apply the local L operator to rho and then apply the
    # conjugate transpose L†
    updated_lk_rho_lkdag = ham_lind.apply_local_op_to_density_matrix(
        density_matrix=rho,
        local_op=lindblad_op,
        target_qubit=target_qubit,
    )

    updated_lk_rho_lkdag = ham_lind.apply_density_matrix_to_local_op_T(
        density_matrix=updated_lk_rho_lkdag,
        local_op=lindblad_op,
        target_qubit=target_qubit,
    )

    listdag = []
    for i in range(nqubits):
        if i == target_qubit:
            listdag.append(lindblad_op.conj().T.contiguous())
        else:
            listdag.append(ident)

    resdag = list_to_kron(listdag)

    assert torch.allclose(updated_lk_rho_lkdag, res @ rho @ resdag)


def test_matmul_linblad_class():
    """Testing 0.5*i*(∑ₖ Lₖ^† Lₖ)@𝜌 + 0.5*i* 𝜌@(∑ₖ Lₖ^† Lₖ) part"""
    torch.manual_seed(234)
    nqubits = 2
    omegas = torch.rand(nqubits, dtype=dtype_adp, device=device).to(dtype)
    deltas = torch.rand(nqubits, dtype=dtype_adp, device=device).to(dtype)
    phis = torch.rand(nqubits, dtype=dtype_adp, device=device).to(dtype)
    pulser_linblads = [
        math.sqrt(1 / 3) * torch.rand(2, 2, dtype=dtype, device="cpu"),  # always on cpu
        math.sqrt(1 / 2) * torch.rand(2, 2, dtype=dtype, device="cpu"),  # always on cpu
    ]
    interaction_matrix = torch.rand(nqubits, nqubits, dtype=dtype, device=device)

    rho = torch.rand(2**nqubits, 2**nqubits, dtype=dtype, device=device)

    ham_lind = RydbergLindbladian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        pulser_linblads=pulser_linblads,
        interaction_matrix=interaction_matrix,
        device=device,
    )
    result_ham = ham_lind @ rho

    # -0.5*i*(∑ₖ Lₖ^† Lₖ) @𝜌 - 0.5*i*𝜌@(∑ₖ Lₖ^† Lₖ) +1j (∑ₖLₖ 𝜌 Lₖ^† )  by hand

    ham = dense_rydberg_hamiltonian(
        omegas=omegas, deltas=deltas, phis=phis, interaction_matrix=interaction_matrix
    )
    h_rho_left = ham @ rho
    h_rho = h_rho_left - h_rho_left.conj().T

    ident = torch.eye(2, dtype=dtype, device=device)

    lista = []
    for lind in pulser_linblads:
        for i in range(nqubits):
            identities = [ident] * nqubits
            lind = lind.to(device)
            identities[i] = lind.conj().T.contiguous() @ lind
            lista.append(identities)

    result_kron_sum_LdagLrho = torch.zeros_like(rho)
    for num, _ in enumerate(lista):
        res = torch.kron(lista[num][0].to(device), lista[num][1].to(device))
        for i in range(2, nqubits):
            res = torch.kron(res, lista[num][i].to(device))
        pre_result = -0.5j * res @ rho
        result_kron_sum_LdagLrho += pre_result - pre_result.conj().T

    # add the term sum Lₖ 𝜌 Lₖ^† to the test

    lista1 = []
    lista2 = []
    for lind in pulser_linblads:
        for i in range(nqubits):
            identities = [ident] * nqubits
            identities[i] = lind.to(device)
            lista1.append(identities.copy())
            identities[i] = (lind.conj().T.contiguous()).to(device)
            lista2.append(identities)

    pre_result = torch.zeros_like(rho)
    for num, _ in enumerate(lista1):
        res1 = list_to_kron(lista1[num])
        res2 = list_to_kron(lista2[num])
        pre_result += 1.0j * res1 @ rho @ res2

    assert torch.allclose(result_ham, h_rho + result_kron_sum_LdagLrho + pre_result)


def test_expect():

    # testing tr(𝜌 H), where H= U₀₁ n₀⊗ n₁ + U₁₂ n₁⊗ n₂ and with U₀₁ = U₁₂ = 1
    # For 3 atoms, H gives diag(0,0,0,1,0,0,1,2), the rest elements are 0
    # expectation is always calculated without jumps opearators
    seed = 1234
    torch.manual_seed(seed)
    nqubits = 3
    omegas = torch.zeros(nqubits, dtype=dtype_adp)
    deltas = torch.zeros(nqubits, dtype=dtype_adp)
    phis = torch.zeros(nqubits, dtype=dtype_adp)

    interaction_mat = nn_interaction_matrix(nqubits)  # U₀₁ n₀⊗ n₁ + U₁₂ n₁⊗ n₂

    pulser_linbdlas = torch.rand(2, 2, dtype=dtype)  # no jump operators
    ham = RydbergLindbladian(
        omegas, deltas, phis, pulser_linbdlas, interaction_mat, device=device
    )
    # creating the density matrix from a random vector as diagonal
    diag_mat = torch.rand(2**nqubits, dtype=dtype_adp, device=device)
    diag_mat = diag_mat / diag_mat.sum()

    densi_mat = DensityMatrix(torch.diag(diag_mat), gpu=gpu)
    # calculating the expectation value
    en = ham.expect(densi_mat)
    expected = (
        diag_mat[3] + diag_mat[-2] + 2 * diag_mat[-1]
    )  # from non zero elements in H

    assert torch.allclose(expected, en)
