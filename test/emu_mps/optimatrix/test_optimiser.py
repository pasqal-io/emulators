import pytest
import random
import numpy as np
import emu_mps.optimatrix.optimiser as optimiser

random.seed(42)


def test_matrix_bandwidth() -> None:
    # Test bandwidth for small matrices 3x3
    def check_bandwidth(matrix: np.ndarray, bandwidth_expected: int) -> None:
        bandwidth = optimiser.matrix_bandwidth(matrix)
        assert optimiser.matrix_bandwidth(matrix.T) == bandwidth  # Transposed
        assert bandwidth == bandwidth_expected

    diagonal_matrix = np.eye(3)
    check_bandwidth(diagonal_matrix, 0)

    one_nondiag = np.array(
        [
            [1, -17, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    check_bandwidth(one_nondiag, 17)

    three_nondiag = np.array(
        [
            [1, -17, 0],
            [9, 1, 0],
            [0, 18, 1],
        ]
    )
    check_bandwidth(three_nondiag, 18)

    matrix_arbitrary = np.array(
        [
            [1, -17, 2.3],
            [9, 1, -10],
            [-15, 20, 1],
        ]
    )
    check_bandwidth(matrix_arbitrary, 30)

    # Test bandwidth for large matrices
    for N in range(10, 100, 10):
        matrix = np.random.rand(N, N)
        i, j = np.random.choice(np.arange(0, N), size=2, replace=False)  # unique ints
        # theoretical bandwidth for np.random.rand(N, N) cannot
        # exceed N - 1 because all matrix elements are in (0,1)

        matrix[i, j] = N  # we replace one element with a big number
        check_bandwidth(matrix, abs(N * (j - i)))


@pytest.mark.parametrize("N", [10, 20, 30])
def test_minimize_bandwidth_global(N: int) -> None:
    # Test shuffled 1D ising chain which is described by tridiagonal matrix {1, 0 , 1}
    mat = np.diag([1] * (N - 1), k=1)
    mat += np.diag([1] * (N - 1), k=-1)
    permuted_mat = optimiser.permute_matrix(mat, list(np.random.permutation(N)))
    optimal_perm = optimiser.minimize_bandwidth_global(permuted_mat)
    assert np.array_equal(mat, optimiser.permute_matrix(permuted_mat, optimal_perm))


@pytest.mark.parametrize("N", [10, 20, 30])
def test_minimize_bandwidth(N: int) -> None:
    # Test sanitizer of symmetric matrices
    mat = np.zeros((N, N))
    mat[0, N - 1] = 1.0  # just a sign to break symmetric condition
    with pytest.raises(AssertionError) as exc_msg:
        optimiser.minimize_bandwidth(mat)
    assert str(exc_msg.value) == "Input matrix is not symmetric"

    def random_permute_matrix(mat: np.ndarray) -> np.ndarray:
        s = mat.shape[0]
        perm_random = random.sample(list(range(s)), s)
        return optimiser.permute_matrix(mat, perm_random)

    # Test 1. The optimal 1D OPEN chain is tridiagonal
    subdiagonal = [1] * (N - 1)
    tridiagonal_matrix = np.diag(subdiagonal, k=1) + np.diag(subdiagonal, k=-1)

    shuffled_matrix = random_permute_matrix(tridiagonal_matrix)
    optimal_perm = optimiser.minimize_bandwidth(shuffled_matrix, samples=10)
    opt_matrix = optimiser.permute_matrix(shuffled_matrix, optimal_perm)
    assert np.array_equal(tridiagonal_matrix, opt_matrix)

    # Test 2. 1D PERIODIC
    subdiagonal = [1] * (N - 1)
    initial_mat = np.diag(subdiagonal, k=1) + np.diag(subdiagonal, k=-1)
    initial_mat[N - 1, 0] = 1
    initial_mat[0, N - 1] = 1

    sub_subdiagonal = [1] * (N - 2)
    expected_mat = np.diag(sub_subdiagonal, k=2) + np.diag(sub_subdiagonal, k=-2)
    expected_mat[1, 0] = expected_mat[0, 1] = 1
    expected_mat[N - 1, N - 2] = expected_mat[N - 2, N - 1] = 1

    optimal_perm = optimiser.minimize_bandwidth(initial_mat, samples=10)
    opt_matrix = optimiser.permute_matrix(initial_mat, optimal_perm)
    assert np.array_equal(expected_mat, opt_matrix)

    shuffled_matrix = random_permute_matrix(initial_mat)
    optimal_perm = optimiser.minimize_bandwidth(shuffled_matrix, samples=10)
    opt_matrix = optimiser.permute_matrix(shuffled_matrix, optimal_perm)
    assert np.array_equal(expected_mat, opt_matrix)


@pytest.mark.parametrize("N", [10, 20, 30])
def test_is_symmetric(N: int) -> None:
    mat = np.zeros((N, N))
    mat[0, N - 1] = 1.0
    assert not optimiser.is_symmetric(mat)

    assert not optimiser.is_symmetric(np.arange(6).reshape((3, 2)))
    assert not optimiser.is_symmetric(np.arange(8).reshape((2, 4)))

    assert optimiser.is_symmetric(mat + mat.T)


def test_2rings_1bar() -> None:
    # ring with 3 qubits, bar with 1 qubit
    input_mat = np.array(
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

    expected_mat = np.array(
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

    optimal_perm = optimiser.minimize_bandwidth(input_mat)
    opt_matrix = optimiser.permute_matrix(input_mat, optimal_perm)
    exp_bandwidth = optimiser.matrix_bandwidth(expected_mat)
    opt_bandwidth = optimiser.matrix_bandwidth(opt_matrix)
    assert exp_bandwidth == opt_bandwidth
