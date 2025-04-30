import math
import pytest
import torch
from emu_sv.lindblad_operator import RydbergLindbladian
from test.utils_testing import dense_rydberg_hamiltonian, nn_interaction_matrix

dtype = torch.complex128
dtype_adp = torch.float64
device = "cpu"


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


test_atoms = 5


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

    res = torch.kron(lista[0], lista[1])
    for i in range(2, nqubits):
        res = torch.kron(res, lista[i])

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

    resa = torch.kron(lista[0], lista[1])
    resdag = torch.kron(listdag[0], listdag[1])
    for i in range(2, nqubits):
        resa = torch.kron(resa, lista[i])
        resdag = torch.kron(resdag, listdag[i])

    assert torch.allclose(updated_lk_rho_lkdag, resa @ rho @ resdag)


# always on cpu
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
            identities[i] = lind
            lista1.append(identities.copy())
            identities[i] = lind.conj().T.contiguous()
            lista2.append(identities)

    pre_result = torch.zeros_like(rho)
    for num, _ in enumerate(lista1):
        res1 = torch.kron(lista1[num][0].to(device), lista1[num][1].to(device))
        res2 = torch.kron(lista2[num][0].to(device), lista2[num][1].to(device))
        for i in range(2, nqubits):
            res1 = torch.kron(res1, lista1[num][i].to(device))
            res2 = torch.kron(res2, lista2[num][i].to(device))
        pre_result += 1.0j * res1 @ rho @ res2

    assert torch.allclose(result_ham, h_rho + result_kron_sum_LdagLrho + pre_result)
