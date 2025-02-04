import pytest
import random
import numpy as np
import optimatrix.optimiser as optimiser

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
    def random_permute_matrix(mat: np.ndarray) -> np.ndarray:
        s = mat.shape[0]
        perm_random = random.sample(list(range(s)), s)
        return optimiser.permute_matrix(mat, perm_random)

    # Test 1. The optimal 1D OPEN chain is tridiagonal
    subdiagonal = [1] * (N - 1)
    tridiagonal_matrix = np.diag(subdiagonal, k=1) + np.diag(subdiagonal, k=-1)

    shuffled_matrix = random_permute_matrix(tridiagonal_matrix)
    optimal_perm = optimiser.minimize_bandwidth(shuffled_matrix, samples = 10)
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

    optimal_perm = optimiser.minimize_bandwidth(initial_mat, samples = 10)
    opt_matrix = optimiser.permute_matrix(initial_mat, optimal_perm)
    assert np.array_equal(expected_mat, opt_matrix)

    shuffled_matrix = random_permute_matrix(initial_mat)
    optimal_perm = optimiser.minimize_bandwidth(shuffled_matrix, samples = 10)
    opt_matrix = optimiser.permute_matrix(shuffled_matrix, optimal_perm)
    assert np.array_equal(expected_mat, opt_matrix)


@pytest.mark.parametrize("N", [10, 20, 30])
def test_is_symmetric(N: int) -> None:
    #Test sanitizer of symmetric matrices
    mat = np.zeros((N, N))
    mat[0, N - 1] = 1.0  # just a sign to break symmetric condition
    with pytest.raises(ValueError) as exc_msg:
        optimiser.is_symmetric(mat)
    assert str(exc_msg.value) == "Input matrix should be symmetric"


    def test_shape(matrix: np.ndarray) -> None:
        msg = f"Input matrix should be square matrix, you provide matrix {matrix.shape}"
        with pytest.raises(ValueError) as exc_msg:
            optimiser.is_symmetric(matrix)
        assert str(exc_msg.value) == msg

    test_shape(np.arange(6).reshape((3, 2)))
    test_shape(np.arange(8).reshape((2, 4)))

def test_2rings_1bar() -> None:
    input_mat = np.array([
        [0.        , 0.28566851, 0.10911566, 0.10911566, 0.28566851, 0.01579137, 0.02027431, 0.03749898, 0.03749898, 0.02027431, 0.25      , 0.0625  ],
        [0.28566851, 0.        , 0.28566851, 0.10911566, 0.10911566, 0.0118583 , 0.01579137, 0.02596986, 0.0226397 , 0.01379494, 0.08403718, 0.03533765],
        [0.10911566, 0.28566851, 0.        , 0.28566851, 0.10911566, 0.00845214, 0.01051012, 0.01579137, 0.01496417, 0.00991961, 0.04051676, 0.02074788],
        [0.10911566, 0.10911566, 0.28566851, 0.        , 0.28566851, 0.00845214, 0.00991961, 0.01496417, 0.01579137, 0.01051012, 0.04051676, 0.02074788],
        [0.28566851, 0.10911566, 0.10911566, 0.28566851, 0.        , 0.0118583 , 0.01379494, 0.0226397 , 0.02596986, 0.01579137, 0.08403718, 0.03533765],
        [0.01579137, 0.0118583 , 0.00845214, 0.00845214, 0.0118583 , 0.        , 0.28566851, 0.10911566, 0.10911566, 0.28566851, 0.02817318, 0.06384162],
        [0.02027431, 0.01579137, 0.01051012, 0.00991961, 0.01379494, 0.28566851, 0.        , 0.28566851, 0.10911566, 0.10911566, 0.0386228 , 0.09560789],
        [0.03749898, 0.02596986, 0.01579137, 0.01496417, 0.0226397 , 0.10911566, 0.28566851, 0.        , 0.28566851, 0.10911566, 0.09659071, 0.49054868],
        [0.03749898, 0.0226397 , 0.01496417, 0.01579137, 0.02596986, 0.10911566, 0.10911566, 0.28566851, 0.        , 0.28566851, 0.09659071, 0.49054868],
        [0.02027431, 0.01379494, 0.00991961, 0.01051012, 0.01579137, 0.28566851, 0.10911566, 0.10911566, 0.28566851, 0.        , 0.0386228 , 0.09560789],
        [0.25      , 0.08403718, 0.04051676, 0.04051676, 0.08403718, 0.02817318, 0.0386228 , 0.09659071, 0.09659071, 0.0386228 , 0.        , 0.25      ],
        [0.0625    , 0.03533765, 0.02074788, 0.02074788, 0.03533765, 0.06384162, 0.09560789, 0.49054868, 0.49054868, 0.09560789, 0.25      , 0.        ],
    ])

    expected_mat = np.array([
        [0.        , 0.49054868, 0.28566851, 0.09659071, 0.10911566, 0.0226397 , 0.01496417, 0.28566851, 0.02596986, 0.10911566, 0.03749898, 0.01579137],
        [0.49054868, 0.        , 0.49054868, 0.25      , 0.09560789, 0.03533765, 0.02074788, 0.09560789, 0.03533765, 0.06384162, 0.0625    , 0.02074788],
        [0.28566851, 0.49054868, 0.        , 0.09659071, 0.28566851, 0.02596986, 0.01579137, 0.10911566, 0.0226397 , 0.10911566, 0.03749898, 0.01496417],
        [0.09659071, 0.25      , 0.09659071, 0.        , 0.0386228 , 0.08403718, 0.04051676, 0.0386228 , 0.08403718, 0.02817318, 0.25      , 0.04051676],
        [0.10911566, 0.09560789, 0.28566851, 0.0386228 , 0.        , 0.01579137, 0.01051012, 0.10911566, 0.01379494, 0.28566851, 0.02027431, 0.00991961],
        [0.0226397 , 0.03533765, 0.02596986, 0.08403718, 0.01579137, 0.        , 0.28566851, 0.01379494, 0.10911566, 0.0118583 , 0.28566851, 0.10911566],
        [0.01496417, 0.02074788, 0.01579137, 0.04051676, 0.01051012, 0.28566851, 0.        , 0.00991961, 0.10911566, 0.00845214, 0.10911566, 0.28566851],
        [0.28566851, 0.09560789, 0.10911566, 0.0386228 , 0.10911566, 0.01379494, 0.00991961, 0.        , 0.01579137, 0.28566851, 0.02027431, 0.01051012],
        [0.02596986, 0.03533765, 0.0226397 , 0.08403718, 0.01379494, 0.10911566, 0.10911566, 0.01579137, 0.        , 0.0118583 , 0.28566851, 0.28566851],
        [0.10911566, 0.06384162, 0.10911566, 0.02817318, 0.28566851, 0.0118583 , 0.00845214, 0.28566851, 0.0118583 , 0.        , 0.01579137, 0.00845214],
        [0.03749898, 0.0625    , 0.03749898, 0.25      , 0.02027431, 0.28566851, 0.10911566, 0.02027431, 0.28566851, 0.01579137, 0.        , 0.10911566],
        [0.01579137, 0.02074788, 0.01496417, 0.04051676, 0.00991961, 0.10911566, 0.28566851, 0.01051012, 0.28566851, 0.00845214, 0.10911566, 0.        ],
    ])

    optimal_perm = optimiser.minimize_bandwidth(input_mat)
    opt_matrix = optimiser.permute_matrix(input_mat, optimal_perm)
    assert np.array_equal(expected_mat, opt_matrix)