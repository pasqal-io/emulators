import torch


def eye_permutation(n: int) -> torch.Tensor:
    """
    Returns toch.tensor([0, 1, 2, .., n-1])
    """
    return torch.arange(n)


def permute_list(input_list: list, perm: torch.Tensor) -> list:
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
    >>> permute_list(['a','b','c'], torch.tensor([2, 0, 1]))
    ['c', 'a', 'b']
    """
    return [input_list[i] for i in perm.tolist()]


def permute_tuple(input_tuple: tuple, perm: torch.Tensor) -> tuple:
    """
    Permutes the input tuple according to the given permutation.
    Parameters
    -------
    input_tuple :
        A tuple to permute.
    permutation :
        A tuple of indices representing the new order.
    Returns
    -------
        The permuted tuple.
    Example
    -------
    >>> permute_tuple(('a','b','c'), torch.tensor([2, 0, 1]))
    ('c', 'a', 'b')
    """
    lst_elem = list(input_tuple)
    return tuple(permute_list(lst_elem, perm))


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
    permuted = permute_list(list(input_str), perm)
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


def permute_tensor(tensor: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """
    Permute a 1D or square 2D torch tensor using the given permutation indices.
    For 1D tensors, applies the permutation to the elements.
    For 2D square tensors, applies the same permutation to both rows and columns.

    Parameters
    ----------
    tensor : torch.Tensor
        A 1D or 2D square tensor to be permuted.
    perm : torch.Tensor
        A 1D tensor of indices specifying the permutation order.

    Returns
    -------
    torch.Tensor
        A new tensor with elements (1D) or rows and columns (2D) permuted according to `perm`.

    Raises
    ------
    ValueError
        If tensor is not 1D or square 2D.

    Examples
    --------
    >>> vector = torch.tensor([10, 20, 30])
    >>> perm = torch.tensor([2, 0, 1])
    >>> permute_tensor(vector, perm)
    tensor([30, 10, 20])

    >>> matrix = torch.tensor([
    ...     [1, 2, 3],
    ...     [4, 5, 6],
    ...     [7, 8, 9]])
    >>> perm = torch.tensor([1, 0, 2])
    >>> permute_tensor(matrix, perm)
    tensor([[5, 4, 6],
            [2, 1, 3],
            [8, 7, 9]])
    """
    if tensor.ndim == 1:
        return tensor[perm]
    elif tensor.ndim == 2 and tensor.shape[0] == tensor.shape[1]:
        return tensor[perm][:, perm]
    else:
        raise ValueError("Only 1D tensors or square 2D tensors are supported.")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
