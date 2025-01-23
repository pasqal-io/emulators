from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import numpy as np
from optimatrix.permutations import permute_matrix, permute_list


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

    if mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"Input matrix should be square matrix, you provide matrix {mat.shape}"
        )
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
    sparse_matrix = csr_matrix(matrix_truncated)  # required for the next line
    rcm_permutation = reverse_cuthill_mckee(sparse_matrix, symmetric_mode=True)
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
    if not np.allclose(mat, mat.T, rtol=1e-8, atol=0):
        raise ValueError("Input matrix should be symmetric")

    mat_amplitude = np.ptp(np.abs(mat).ravel())  # mat.abs.max - mat.abs().min()

    # Search from 1.0 to 0.1 doesn't change result
    permutations = (
        minimize_bandwidth_above_threshold(mat, trunc * mat_amplitude)
        for trunc in np.arange(start=0.1, stop=1.0, step=0.01)
    )

    opt_permutation = min(
        permutations, key=lambda perm: matrix_bandwidth(permute_matrix(mat, list(perm)))
    )
    return list(opt_permutation)  # opt_permutation is np.ndarray


def minimize_bandwidth(matrix: np.ndarray) -> list[int]:
    """
    minimize_bandwidth(matrix) -> list

    Finds the permutation list for a symmetric matrix that iteratively minimizes matrix bandwidth.

    Parameters
    -------
    matrix :
        symmetric square matrix

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
    >>> minimize_bandwidth(matrix) # [3, 2, 4, 1, 0] does zig-zag
    [3, 2, 4, 1, 0]

    Simple 1D chain. Cannot be optimised further
    >>> matrix = np.array([
    ...    [0, 1, 0, 0, 0],
    ...    [1, 0, 1, 0, 0],
    ...    [0, 1, 0, 1, 0],
    ...    [0, 0, 1, 0, 1],
    ...    [0, 0, 0, 1, 0]])
    >>> minimize_bandwidth(matrix)
    [0, 1, 2, 3, 4]
    """
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError("Input matrix should be symmetric")
    mat = abs(
        matrix.copy()
    )  # sanitizer for cuthill-mckee. We are interested in strength of the interaction, not sign

    acc_permutation = list(
        range(matrix.shape[0])
    )  # start with trivial permutation [0, 1, 2, ...]

    bandwidth = matrix_bandwidth(mat)

    counter = 100
    while True:
        if counter < 0:
            raise (
                NotImplementedError(
                    "The algorithm takes too many steps, " "probably not converging."
                )
            )
        counter -= 1

        optimal_perm = minimize_bandwidth_global(mat)
        test_mat = permute_matrix(mat, optimal_perm)
        new_bandwidth = matrix_bandwidth(test_mat)

        if bandwidth <= new_bandwidth:
            break

        mat = test_mat
        acc_permutation = permute_list(acc_permutation, optimal_perm)
        bandwidth = new_bandwidth

    return acc_permutation


if __name__ == "__main__":
    import doctest

    doctest.testmod()
