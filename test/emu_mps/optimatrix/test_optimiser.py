import pytest

import torch
import emu_mps.optimatrix.optimiser as opt

torch.manual_seed(42)


def test_matrix_bandwidth() -> None:
    # Test bandwidth for small matrices 3x3
    def check_bandwidth(matrix: torch.Tensor, bandwidth_expected: int) -> None:
        bandwidth = opt.matrix_bandwidth(matrix)
        assert opt.matrix_bandwidth(matrix.T) == bandwidth  # Transposed
        assert bandwidth == bandwidth_expected

    diagonal_matrix = torch.eye(3)
    check_bandwidth(diagonal_matrix, 0)

    one_nondiag = torch.tensor(
        [
            [1, -17, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    check_bandwidth(one_nondiag, 17)

    three_nondiag = torch.tensor(
        [
            [1, -17, 0],
            [9, 1, 0],
            [0, 18, 1],
        ]
    )
    check_bandwidth(three_nondiag, 18)

    matrix_arbitrary = torch.tensor(
        [
            [1, -17, 2.3],
            [9, 1, -10],
            [-15, 20, 1],
        ]
    )
    check_bandwidth(matrix_arbitrary, 30)


def test_minimize_bandwidth_sanitizer() -> None:
    N = 4
    mat = torch.zeros((N, N))
    mat[0, N - 1] = 1.0  # to break symmetric condition
    with pytest.raises(AssertionError) as exc_msg:
        opt.minimize_bandwidth(mat)
    assert str(exc_msg.value) == "Input matrix is not symmetric"


@pytest.mark.parametrize("N", [4, 5])
def test_OBC_minimize_bandwidth(N: int) -> None:
    # Open Boundary Condition == 1D chain
    upper = torch.diag(torch.ones(N - 1), diagonal=1)
    lower = torch.diag(torch.ones(N - 1), diagonal=-1)
    tridiag_mat = upper + lower

    shuffled_mat = opt.permute_tensor(tridiag_mat, torch.randperm(N))
    optimal_perm = opt.minimize_bandwidth(shuffled_mat, samples=10)

    opt_matrix = opt.permute_tensor(shuffled_mat, optimal_perm)
    assert torch.equal(tridiag_mat, opt_matrix)


@pytest.mark.parametrize("N", [4, 5])
def test_PBC_minimize_bandwidth(N: int) -> None:
    # Periodic Boundary Condition == 1D ring
    upper = torch.diag(torch.ones(N - 1), diagonal=1)
    lower = torch.diag(torch.ones(N - 1), diagonal=-1)
    initial_mat = upper + lower
    initial_mat[N - 1, 0] = initial_mat[0, N - 1] = 1

    expected_mat = torch.diag(torch.ones(N - 2), diagonal=2) + torch.diag(
        torch.ones(N - 2), diagonal=-2
    )
    expected_mat[1, 0] = expected_mat[0, 1] = 1
    expected_mat[N - 1, N - 2] = expected_mat[N - 2, N - 1] = 1

    optimal_perm = opt.minimize_bandwidth(initial_mat, samples=10)
    opt_matrix = opt.permute_tensor(initial_mat, optimal_perm)
    assert torch.equal(expected_mat, opt_matrix)


def test_is_symmetric() -> None:
    N = 3
    mat = torch.zeros((N, N))
    mat[0, N - 1] = 1.0
    assert not opt.is_symmetric(mat)
    assert opt.is_symmetric(mat + mat.T)


def test_2rings_1bar() -> None:
    # 2 rings with 3 qubits, bar with 1 qubit = 7 qubits
    input_mat = torch.tensor(
        [
            [0.0, 0.3655409, 0.3655409, 0.04386491, 0.08435559, 0.08435559, 0.25],
            [0.3655409, 0.0, 0.3655409, 0.02550285, 0.04386491, 0.0391651, 0.08022302],
            [0.3655409, 0.3655409, 0.0, 0.02550285, 0.0391651, 0.04386491, 0.08022302],
            [0.04386491, 0.02550285, 0.02550285, 0.0, 0.3655409, 0.3655409, 0.12989251],
            [0.08435559, 0.04386491, 0.0391651, 0.3655409, 0.0, 0.3655409, 0.40232329],
            [0.08435559, 0.0391651, 0.04386491, 0.3655409, 0.3655409, 0.0, 0.40232329],
            [0.25, 0.08022302, 0.08022302, 0.12989251, 0.40232329, 0.40232329, 0.0],
        ]
    )

    expected_mat = torch.tensor(
        [
            [0.0, 0.3655409, 0.3655409, 0.12989251, 0.04386491, 0.02550285, 0.02550285],
            [0.3655409, 0.0, 0.3655409, 0.40232329, 0.08435559, 0.0391651, 0.04386491],
            [0.3655409, 0.3655409, 0.0, 0.40232329, 0.08435559, 0.04386491, 0.0391651],
            [0.12989251, 0.40232329, 0.40232329, 0.0, 0.25, 0.08022302, 0.08022302],
            [0.04386491, 0.08435559, 0.08435559, 0.25, 0.0, 0.3655409, 0.3655409],
            [0.02550285, 0.0391651, 0.04386491, 0.08022302, 0.3655409, 0.0, 0.3655409],
            [0.02550285, 0.04386491, 0.0391651, 0.08022302, 0.3655409, 0.3655409, 0.0],
        ]
    )

    optimal_perm = opt.minimize_bandwidth(input_mat)
    opt_matrix = opt.permute_tensor(input_mat, optimal_perm)
    exp_bandwidth = opt.matrix_bandwidth(expected_mat)
    opt_bandwidth = opt.matrix_bandwidth(opt_matrix)
    assert exp_bandwidth == opt_bandwidth
