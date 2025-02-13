import numpy as np


def permute_list(input_list: list, permutation: list[int]) -> list:
    """
    Permutes the input list according to the given permutation.

    Parameters
    -------
    input_list :
        A list to permute.
    permutation :
        A list of indices representing the new order.

    Returns
    -------
        The permuted list.

    Example
    -------
    >>> permute_list(['a', 'b', 'c'], [2, 0, 1])
    ['c', 'a', 'b']
    """

    permuted_list = [None] * len(input_list)
    for i, p in enumerate(permutation):
        permuted_list[i] = input_list[p]
    return permuted_list


def invert_permutation(permutation: list[int]) -> list[int]:
    """
    invert_permutation(permutation) -> inv_permutation

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
    >>> invert_permutation([2, 0, 1])
    [1, 2, 0]
    """

    inv_perm = np.empty_like(permutation)
    inv_perm[permutation] = np.arange(len(permutation))
    return list(inv_perm)


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
