import torch
from emu_sv.lindblad_operator import LindbladOperator
from test.utils_testing import dense_rydberg_hamiltonian, nn_interaction_matrix

dtype = torch.complex128
device = torch.device("cpu")


def test_linblad():
    torch.manual_seed(234)
    nqubits = 3
    omegas = torch.tensor([1.0, 2.0, 4.0], dtype=dtype, device=device)
    deltas = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)
    phis = torch.zeros(nqubits, dtype=dtype, device=device)
    pulser_linblad = torch.zeros(2**nqubits, dtype=dtype, device=device)
    interaction_matrix = nn_interaction_matrix(nqubits)

    rho = torch.rand(2**nqubits, 2**nqubits, dtype=dtype, device=device)
    ham_lind = LindbladOperator(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        pulser_linblad=pulser_linblad,
        interaction_matrix=interaction_matrix,
        device=device,
    )
    result = ham_lind @ rho

    ham = dense_rydberg_hamiltonian(
        omegas=omegas, deltas=deltas, phis=phis, interaction_matrix=interaction_matrix
    )
    h_rho = ham @ rho

    assert torch.allclose(result, h_rho, atol=1e-5), "LindbladOperator test failed"
