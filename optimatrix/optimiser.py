from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import numpy as np
from optimatrix.permutations import permute_matrix, permute_list


def is_symmetric(mat: np.ndarray) -> None:
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"Input matrix should be square matrix, you provide matrix {mat.shape}"
        )
    if not np.allclose(mat, mat.T, atol=1e-8):
        raise ValueError("Input matrix should be symmetric")
    
    return None


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
    rcm_permutation = reverse_cuthill_mckee(csr_matrix(matrix_truncated), symmetric_mode=True)
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
    mat_amplitude = np.max(np.abs(mat))#np.ptp(np.abs(mat).ravel())  # mat.abs.max - mat.abs().min()

    # Search from 1.0 to 0.1 doesn't change result
    # Search from 0.1 to 1.0 allows to remove copying from minimize_bandwidth_above_threshold
    permutations = (
        minimize_bandwidth_above_threshold(mat, trunc * mat_amplitude)
        for trunc in np.arange(start=0.1, stop=1.0, step=0.01)
    )

    opt_permutation = min(
        permutations, key=lambda perm: matrix_bandwidth(permute_matrix(mat, list(perm)))
    )
    return list(opt_permutation)  # opt_permutation is np.ndarray


def minimize_bandwidth_impl(matrix: np.ndarray) -> list[int]:
    """
    minimize_bandwidth_impl(matrix) -> list

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
    >>> minimize_bandwidth_impl(matrix) # [3, 2, 4, 1, 0] does zig-zag
    [3, 2, 4, 1, 0]

    Simple 1D chain. Cannot be optimised further
    >>> matrix = np.array([
    ...    [0, 1, 0, 0, 0],
    ...    [1, 0, 1, 0, 0],
    ...    [0, 1, 0, 1, 0],
    ...    [0, 0, 1, 0, 1],
    ...    [0, 0, 0, 1, 0]])
    >>> minimize_bandwidth_impl(matrix)
    [0, 1, 2, 3, 4]
    """
    acc_permutation = list(
        range(matrix.shape[0])
    )  # start with trivial permutation [0, 1, 2, ...]
    bandwidth = matrix_bandwidth(matrix)


    for counter in range(101):
        if counter == 100:
            raise (
                NotImplementedError(
                    "The algorithm takes too many steps, " "probably not converging."
                )
            )

        optimal_perm = minimize_bandwidth_global(matrix.copy())  # modifies the matrix
        test_mat = permute_matrix(matrix, optimal_perm)
        new_bandwidth = matrix_bandwidth(test_mat)

        if bandwidth <= new_bandwidth:
            break

        matrix = test_mat
        acc_permutation = permute_list(acc_permutation, optimal_perm)
        bandwidth = new_bandwidth

    return acc_permutation


def minimize_bandwidth(input_mat: np.ndarray, samples: int = 100) -> list[int]:
    is_symmetric(input_mat)
    input_mat = abs(input_mat)
    # We are interested in strength of the interaction, not sign

    L = input_mat.shape[0]
    rnd_permutations = [
        np.random.permutation(L).tolist() for _ in range(samples)
    ] #rnd samples cannot be generator
    rnd_permutations.insert(0, list(range(L))) #initial non-randomized order

    opt_permutations = (
        minimize_bandwidth_impl(permute_matrix(input_mat, rnd_perm))
        for rnd_perm in rnd_permutations
    )
    best_permutations = (
        permute_list(rnd_perm, opt_perm)
        for rnd_perm, opt_perm in zip(rnd_permutations, opt_permutations)
    )
    return min(best_permutations, key = lambda perm: matrix_bandwidth(permute_matrix(input_mat, perm)))






if __name__ == "__main__":
    import doctest

    doctest.testmod()
