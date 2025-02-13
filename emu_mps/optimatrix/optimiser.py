from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import numpy as np
from emu_mps.optimatrix.permutations import permute_matrix, permute_list
import itertools


def is_symmetric(mat: np.ndarray) -> bool:
    if mat.shape[0] != mat.shape[1]:
        return False
    if not np.allclose(mat, mat.T, atol=1e-8):
        return False

    return True


def matrix_bandwidth(mat: np.ndarray) -> float:
    """matrix_bandwidth(matrix: np.ndarray) -> float

    Computes bandwidth as max weighted distance between columns of
    a square matrix as `max (abs(matrix[i, j] * (j - i))`.

             abs(j-i)
          |<--------->|
        (i,i)       (i,j)
          |           |
    | *   .   .   .   .   . |
    | .   *   .   .   a   . |
    | .   .   *   .   .   . |
    | .   .   .   *   .   . |
    | .   .   .   .   *   . |
    | .   .   .   .   .   * |

    Distance from the main diagonal `[i,i]` and element `m[i,j]` along row is
    `abs(j-i)` and therefore the weighted distance is `abs(matrix[i, j] * (j - i))`

    Parameters
    -------
    matrix :
        square matrix nxn

    Returns
    -------
        bandwidth of the input matrix

    Example:
    -------
    >>> matrix = np.array([
    ...    [  1, -17, 2.4],
    ...    [  9,   1, -10],
    ...    [-15,  20,   1],])
    >>> matrix_bandwidth(matrix) # 30.0 because abs(-15 * (2-0) == 30)
    30.0
    """

    bandwidth = max(abs(el * (index[0] - index[1])) for index, el in np.ndenumerate(mat))
    return float(bandwidth)


def minimize_bandwidth_above_threshold(mat: np.ndarray, threshold: float) -> np.ndarray:
    """
    minimize_bandwidth_above_threshold(matrix, trunc) -> permutation_lists

    Finds a permutation list that minimizes a bandwidth of a symmetric matrix `A = A.T`
    using the reverse Cuthill-Mckee algorithm from `scipy.sparse.csgraph.reverse_cuthill_mckee`.
    Matrix elements below a threshold `m[i,j] < threshold` are considered as 0.

    Parameters
    -------
    matrix :
        symmetric square matrix
    threshold :
        matrix elements `m[i,j] < threshold` are considered as 0

    Returns
    -------
        permutation list that minimizes matrix bandwidth for a given threshold

    Example:
    -------
    >>> matrix = np.array([
    ...    [1, 2, 3],
    ...    [2, 5, 6],
    ...    [3, 6, 9]])
    >>> threshold = 3
    >>> minimize_bandwidth_above_threshold(matrix, threshold)
    array([1, 2, 0], dtype=int32)
    """

    matrix_truncated = mat.copy()
    matrix_truncated[mat < threshold] = 0
    rcm_permutation = reverse_cuthill_mckee(
        csr_matrix(matrix_truncated), symmetric_mode=True
    )
    return np.array(rcm_permutation)


def minimize_bandwidth_global(mat: np.ndarray) -> list[int]:
    """
    minimize_bandwidth_global(matrix) -> list

    Does one optimisation step towards finding
    a permutation of a matrix that minimizes matrix bandwidth.

    Parameters
    -------
    matrix :
        symmetric square matrix

    Returns
    -------
        permutation order that minimizes matrix bandwidth

    Example:
    -------
    >>> matrix = np.array([
    ...    [1, 2, 3],
    ...    [2, 5, 6],
    ...    [3, 6, 9]])
    >>> minimize_bandwidth_global(matrix)
    [2, 1, 0]
    """
    mat_amplitude = np.max(np.abs(mat))
    # Search from 1.0 to 0.1 doesn't change result
    permutations = (
        minimize_bandwidth_above_threshold(mat, trunc * mat_amplitude)
        for trunc in np.arange(start=0.1, stop=1.0, step=0.01)
    )

    opt_permutation = min(
        permutations, key=lambda perm: matrix_bandwidth(permute_matrix(mat, list(perm)))
    )
    return list(opt_permutation)  # opt_permutation is np.ndarray


def minimize_bandwidth_impl(
    matrix: np.ndarray, initial_perm: list[int]
) -> tuple[list[int], float]:
    """
    minimize_bandwidth_impl(matrix, initial_perm) -> list

    Applies initial_perm to a matrix and
    finds the permutation list for a symmetric matrix that iteratively minimizes matrix bandwidth.

    Parameters
    -------
    matrix :
        symmetric square matrix
    initial_perm: list of integers


    Returns
    -------
        permutation order that minimizes matrix bandwidth

    Example:
    -------
    Periodic 1D chain
    >>> matrix = np.array([
    ...    [0, 1, 0, 0, 1],
    ...    [1, 0, 1, 0, 0],
    ...    [0, 1, 0, 1, 0],
    ...    [0, 0, 1, 0, 1],
    ...    [1, 0, 0, 1, 0]])
    >>> id_perm = list(range(matrix.shape[0]))
    >>> minimize_bandwidth_impl(matrix, id_perm) # [3, 2, 4, 1, 0] does zig-zag
    ([3, 2, 4, 1, 0], 2.0)

    Simple 1D chain. Cannot be optimised further
    >>> matrix = np.array([
    ...    [0, 1, 0, 0, 0],
    ...    [1, 0, 1, 0, 0],
    ...    [0, 1, 0, 1, 0],
    ...    [0, 0, 1, 0, 1],
    ...    [0, 0, 0, 1, 0]])
    >>> id_perm = list(range(matrix.shape[0]))
    >>> minimize_bandwidth_impl(matrix, id_perm)
    ([0, 1, 2, 3, 4], 1.0)
    """
    if initial_perm != list(range(matrix.shape[0])):
        matrix = permute_matrix(matrix, initial_perm)
    bandwidth = matrix_bandwidth(matrix)
    acc_permutation = initial_perm

    for counter in range(101):
        if counter == 100:
            raise (
                NotImplementedError(
                    "The algorithm takes too many steps, " "probably not converging."
                )
            )

        optimal_perm = minimize_bandwidth_global(matrix)
        test_mat = permute_matrix(matrix, optimal_perm)
        new_bandwidth = matrix_bandwidth(test_mat)

        if bandwidth <= new_bandwidth:
            break

        matrix = test_mat
        acc_permutation = permute_list(acc_permutation, optimal_perm)
        bandwidth = new_bandwidth

    return acc_permutation, bandwidth


def minimize_bandwidth(input_matrix: np.ndarray, samples: int = 100) -> list[int]:
    assert is_symmetric(input_matrix), "Input matrix is not symmetric"
    input_mat = abs(input_matrix)
    # We are interested in strength of the interaction, not sign

    L = input_mat.shape[0]
    rnd_permutations = itertools.chain(
        [list(range(L))],  # First element is always the identity list
        (np.random.permutation(L).tolist() for _ in range(samples)),
    )

    opt_permutations_and_opt_bandwidth = (
        minimize_bandwidth_impl(input_mat, rnd_perm) for rnd_perm in rnd_permutations
    )

    best_perm, best_bandwidth = min(
        opt_permutations_and_opt_bandwidth,
        key=lambda perm_and_bandwidth: perm_and_bandwidth[1],
    )

    assert best_bandwidth <= matrix_bandwidth(input_matrix), "Matrix is not optimised"
    return best_perm


if __name__ == "__main__":
    import doctest

    doctest.testmod()
