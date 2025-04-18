import pytest
import numpy as np
import torch

from emu_mps.optimatrix.permutations import (
    permute_string,
    inv_permutation,
    permute_matrix,
)


def test_permute_string() -> None:
    perm = torch.tensor([1, 0])
    input_str = "ab"
    assert permute_string(input_str, perm) == "ba"

    # 6 permutations of abs
    input_str = "abc"
    assert permute_string(input_str, torch.tensor([0, 1, 2])) == "abc"
    assert permute_string(input_str, torch.tensor([0, 2, 1])) == "acb"
    assert permute_string(input_str, torch.tensor([1, 0, 2])) == "bac"
    assert permute_string(input_str, torch.tensor([1, 2, 0])) == "bca"
    assert permute_string(input_str, torch.tensor([2, 0, 1])) == "cab"
    assert permute_string(input_str, torch.tensor([2, 1, 0])) == "cba"


def test_invert_permutation() -> None:
    # basic tests
    assert torch.all(inv_permutation(torch.tensor([0, 1, 2])) == torch.tensor([0, 1, 2]))
    assert torch.all(inv_permutation(torch.tensor([0, 2, 1])) == torch.tensor([0, 2, 1]))
    assert torch.all(inv_permutation(torch.tensor([1, 0, 2])) == torch.tensor([1, 0, 2]))
    assert torch.all(inv_permutation(torch.tensor([1, 2, 0])) == torch.tensor([2, 0, 1]))
    assert torch.all(inv_permutation(torch.tensor([2, 0, 1])) == torch.tensor([1, 2, 0]))
    assert torch.all(inv_permutation(torch.tensor([2, 1, 0])) == torch.tensor([2, 1, 0]))

    assert torch.all(
        inv_permutation(torch.tensor([2, 1, 3, 0])) == torch.tensor([3, 1, 0, 2])
    )


@pytest.mark.parametrize("N", [10, 20, 30])
def test_implementation_permute_matrix(N: int) -> None:
    perm = np.random.permutation(N)
    rnd_matrix = np.random.rand(N, N)
    rnd_permuted = permute_matrix(rnd_matrix, list(perm))

    assert np.array_equal(rnd_matrix[perm, :][:, perm], rnd_permuted)  # implementation 1
    assert np.array_equal(rnd_matrix[:, perm][perm, :], rnd_permuted)  # implementation 2

    P = np.eye(N)[perm, :]
    assert np.array_equal(P @ rnd_matrix @ P.T, rnd_permuted)  # implementation 3

    P2 = np.eye(N)[:, perm]
    assert np.array_equal(P2.T @ rnd_matrix @ P2, rnd_permuted)  # implementation 4


@pytest.mark.parametrize("N", [10, 20, 30])
def test_permute_matrix(N: int) -> None:
    mat012 = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    assert np.array_equal(
        mat012, permute_matrix(mat012, [0, 1, 2])
    )  # identity permutation

    rnd_matrix = np.random.rand(N, N)
    assert np.array_equal(
        rnd_matrix, permute_matrix(rnd_matrix, list(range(N)))
    )  # identity permutation for large matrices

    # composition of permutations == permutation
    perm1 = np.random.permutation(N)
    perm2 = np.random.permutation(N)
    double_perm = [perm1[i] for i in perm2]

    mat_single_perm = permute_matrix(rnd_matrix, list(perm1))
    mat_double_perm = permute_matrix(mat_single_perm, list(perm2))

    assert np.array_equal(mat_double_perm, permute_matrix(rnd_matrix, double_perm))
