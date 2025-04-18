import numpy as np
import torch


def permute_string(input_str: str, perm: torch.Tensor) -> str:
    """
    Permutes the input string according to the given permutation.
    Parameters
    -------
    input_string :
        A string to permute.
    permutation :
        A list of indices representing the new order.
    Returns
    -------
        The permuted string.
    Example
    -------
    >>> permute_string("abc", torch.tensor([2, 0, 1]))
    'cab'
    """
    char_list = list(input_str)
    permuted = [char_list[i] for i in perm.tolist()]
    return "".join(permuted)


def inv_permutation(permutation: torch.Tensor) -> torch.Tensor:
    """
    inv_permutation(permutation) -> inverted_perm

    Inverts the input permutation list.

    Parameters
    -------
    permutation :
        A list of indices representing the order

    Returns
    -------
        permutation list inverse to the input list

    Example:
    -------
    >>> inv_permutation(torch.tensor([2, 0, 1]))
    tensor([1, 2, 0])
    """
    inv_perm = torch.empty_like(permutation)
    inv_perm[permutation] = torch.arange(len(permutation))
    return inv_perm


def permute_matrix(mat: np.ndarray, permutation: list[int]) -> np.ndarray:
    """
    permute_matrix(matrix, permutation_list) -> permuted_matrix

    Simultaneously permutes columns and rows according to a permutation list.

    Parameters
    -------
    matrix :
        square matrix nxn
    permutation :
        permutation list

    Returns
    -------
        matrix with permuted columns and rows

    Example:
    -------
    >>> matrix = np.array([
    ...    [1, 2, 3],
    ...    [4, 5, 6],
    ...    [7, 8, 9]])
    >>> permutation = [1, 0, 2]
    >>> permute_matrix(matrix, permutation)
    array([[5, 4, 6],
           [2, 1, 3],
           [8, 7, 9]])
    """

    perm = np.array(permutation)
    return mat[perm, :][:, perm]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
