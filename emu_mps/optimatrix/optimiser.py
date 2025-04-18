import itertools

import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from emu_mps.optimatrix.permutations import permute_tensor


def is_symmetric(matrix: torch.Tensor, tol: float = 1e-8) -> bool:
    return torch.allclose(matrix, matrix.T, atol=tol)


def matrix_bandwidth(mat: torch.Tensor) -> float:
    """matrix_bandwidth(matrix: torch.tensor) -> torch.Tensor

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
    >>> matrix = torch.tensor([
    ...     [1.0, -17.0, 2.4],
    ...     [9.0, 1.0, -10.0],
    ...     [-15.0, 20.0, 1.0]
    ... ])
    >>> matrix_bandwidth(matrix)  # because abs(-15 * (0 - 2)) = 30.0
    30.0
    """

    n = mat.shape[0]

    i_arr = torch.arange(n).view(-1, 1)  # shape (n, 1)
    j_arr = torch.arange(n).view(1, -1)  # shape (1, n)

    weighted = torch.abs(mat * (j_arr - i_arr))
    return torch.max(weighted).to(mat.dtype).item()


def minimize_bandwidth_above_threshold(
    mat: torch.Tensor, threshold: float
) -> torch.Tensor:
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
    >>> matrix = torch.tensor([
    ...     [1, 2, 3],
    ...     [2, 5, 6],
    ...     [3, 6, 9]
    ... ], dtype=torch.float32)
    >>> threshold = 3
    >>> minimize_bandwidth_above_threshold(matrix, threshold)
    tensor([1, 2, 0], dtype=torch.int32)
    """

    m_trunc = mat.clone()
    m_trunc[mat < threshold] = 0.0

    matrix_np = csr_matrix(m_trunc.numpy())  # SciPy's RCM compatibility
    rcm_perm = reverse_cuthill_mckee(matrix_np, symmetric_mode=True)
    return torch.from_numpy(rcm_perm.copy())  # translation requires copy


def minimize_bandwidth_global(mat: torch.Tensor) -> torch.Tensor:
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

    Example
    -------
    >>> matrix = torch.tensor([
    ...     [1, 2, 3],
    ...     [2, 5, 6],
    ...     [3, 6, 9]
    ... ], dtype=torch.float32)
    >>> minimize_bandwidth_global(matrix)
    tensor([2, 1, 0], dtype=torch.int32)
    """
    mat_amplitude = torch.max(torch.abs(mat))

    permutations = (
        minimize_bandwidth_above_threshold(mat, trunc.item() * mat_amplitude)
        for trunc in torch.arange(0.1, 1.0, 0.01)
    )

    opt_permutation = min(
        permutations, key=lambda perm: matrix_bandwidth(permute_tensor(mat, perm))
    )

    return opt_permutation


def minimize_bandwidth_impl(
    matrix: torch.Tensor, initial_perm: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """
    minimize_bandwidth_impl(matrix, initial_perm) -> (optimal_perm, bandwidth)

    Applies initial_perm to a matrix and
    finds the permutation list for a symmetric matrix
    that iteratively minimizes matrix bandwidth.

    Parameters
    -------
    matrix :
        symmetric square matrix
    initial_perm: torch list of integers


    Returns
    -------
        optimal permutation and optimal matrix bandwidth

    Example:
    -------
    Periodic 1D chain
    >>> matrix = torch.tensor([
    ...    [0, 1, 0, 0, 1],
    ...    [1, 0, 1, 0, 0],
    ...    [0, 1, 0, 1, 0],
    ...    [0, 0, 1, 0, 1],
    ...    [1, 0, 0, 1, 0]], dtype=torch.float32)
    >>> id_perm = torch.arange(matrix.shape[0])
    >>> minimize_bandwidth_impl(matrix, id_perm) # [3, 2, 4, 1, 0] does zig-zag
    (tensor([3, 2, 4, 1, 0]), 2.0)

    Simple 1D chain. Cannot be optimised further
    >>> matrix = torch.tensor([
    ...    [0, 1, 0, 0, 0],
    ...    [1, 0, 1, 0, 0],
    ...    [0, 1, 0, 1, 0],
    ...    [0, 0, 1, 0, 1],
    ...    [0, 0, 0, 1, 0]], dtype=torch.float32)
    >>> id_perm = torch.arange(matrix.shape[0])
    >>> minimize_bandwidth_impl(matrix, id_perm)
    (tensor([0, 1, 2, 3, 4]), 1.0)
    """
    L = matrix.shape[0]
    if not torch.equal(initial_perm, torch.arange(L)):
        matrix = permute_tensor(matrix, initial_perm)
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
        test_mat = permute_tensor(matrix, optimal_perm)
        new_bandwidth = matrix_bandwidth(test_mat)

        if bandwidth <= new_bandwidth:
            break

        matrix = test_mat
        acc_permutation = permute_tensor(acc_permutation, optimal_perm)
        bandwidth = new_bandwidth

    return acc_permutation, bandwidth


def minimize_bandwidth(input_matrix: torch.Tensor, samples: int = 100) -> torch.Tensor:
    assert is_symmetric(input_matrix), "Input matrix is not symmetric"
    input_mat = torch.abs(input_matrix)
    # We are interested in strength of the interaction, not sign

    L = input_mat.shape[0]
    rnd_permutations = itertools.chain(
        [torch.arange(L)],  # identity permutation
        [torch.randperm(L) for _ in range(samples)],  # list of random permutations
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
