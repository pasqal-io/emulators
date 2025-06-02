import torch
from emu_mps.mps_config import MPSConfig
from emu_mps.dmrg import minimize_energy_pair

dtype = torch.complex128


def test_minimize_energy_pair():
    left_bath = torch.rand(2, 2, 2, dtype=dtype)
    right_bath = torch.rand(2, 3, 2, dtype=dtype)
    left_state_factor = torch.rand(2, 2, 4, dtype=dtype)
    right_state_factor = torch.rand(4, 2, 2, dtype=dtype)
    left_ham_factor = torch.rand(2, 2, 2, 5, dtype=dtype)
    right_ham_factor = torch.rand(5, 2, 2, 3, dtype=dtype)

    # Make the baths and ham factors hermitian tensors over the appropriate indices
    left_bath = 0.5 * (left_bath + left_bath.permute(2, 1, 0).conj())
    right_bath = 0.5 * (right_bath + right_bath.permute(2, 1, 0).conj())
    left_ham_factor = 0.5 * (left_ham_factor + left_ham_factor.permute(0, 2, 1, 3).conj())
    right_ham_factor = 0.5 * (
        right_ham_factor + right_ham_factor.permute(0, 2, 1, 3).conj()
    )

    left_state_factor_shape = left_state_factor.shape
    right_state_factor_shape = right_state_factor.shape

    op = (
        torch.einsum(
            "abc,bdef,fghi,jik->adgjcehk",
            left_bath,
            left_ham_factor,
            right_ham_factor,
            right_bath,
        )
        .contiguous()
        .view(2 * 2 * 2 * 2, -1)
    )

    eigen_energy, _ = torch.linalg.eigh(op)
    expected_energy = eigen_energy[0].item()

    actual_left, actual_right, actual_energy = minimize_energy_pair(
        state_factors=[left_state_factor, right_state_factor],
        baths=[left_bath, right_bath],
        ham_factors=[left_ham_factor, right_ham_factor],
        config=MPSConfig(max_bond_dim=12),
        is_hermitian=True,
        orth_center_right=False,
        residual_tolerance=1e-8,
    )

    # check that resulting state tensors have correct shapes
    assert actual_left.shape == left_state_factor_shape
    assert actual_right.shape == right_state_factor_shape

    # check: norm of new state ~ 1
    actual_state = torch.tensordot(actual_left, actual_right, dims=1).view(-1)
    assert torch.isclose(actual_state.norm(), torch.tensor(1.0, dtype=float), atol=1e-8)

    # check the resulting energy
    assert abs(actual_energy.real - expected_energy.real) <= 1e-8
