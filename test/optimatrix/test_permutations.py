import pytest
import numpy as np
from optimatrix.permutations import permute_list, invert_permutation, permute_matrix
import random


def test_permute_list() -> None:
    perm = [1, 0]
    input_list = ["a", "b"]
    assert permute_list(input_list, perm) == ["b", "a"]

    # 6 permutations of [a, b, c]
    input_list = ["a", "b", "c"]
    assert permute_list(input_list, [0, 1, 2]) == ["a", "b", "c"]
    assert permute_list(input_list, [0, 2, 1]) == ["a", "c", "b"]
    assert permute_list(input_list, [1, 0, 2]) == ["b", "a", "c"]
    assert permute_list(input_list, [1, 2, 0]) == ["b", "c", "a"]
    assert permute_list(input_list, [2, 0, 1]) == ["c", "a", "b"]
    assert permute_list(input_list, [2, 1, 0]) == ["c", "b", "a"]


@pytest.mark.parametrize("N", [10, 20, 30])
def test_permute_list_implementation(N: int) -> None:
    input_list = np.random.permutation(N)
    perm = np.random.permutation(N)

    numpy_expected = list(input_list[perm])  # numpy implementation of permutations
    optimatrix_imp = permute_list(list(input_list), list(perm))
    assert optimatrix_imp == numpy_expected


def test_invert_permutation() -> None:
    # basic tests
    assert invert_permutation([0, 1, 2]) == [0, 1, 2]
    assert invert_permutation([0, 2, 1]) == [0, 2, 1]
    assert invert_permutation([1, 0, 2]) == [1, 0, 2]
    assert invert_permutation([1, 2, 0]) == [2, 0, 1]
    assert invert_permutation([2, 0, 1]) == [1, 2, 0]
    assert invert_permutation([2, 1, 0]) == [2, 1, 0]

    assert invert_permutation([2, 1, 3, 0]) == [3, 1, 0, 2]


@pytest.mark.parametrize("N", [10, 20, 30])
def test_invert_permutation_random(N: int) -> None:
    # inverse of arbitrary permutation gives the initial list
    perm_random = random.sample(list(range(N)), N)
    inv_perm = invert_permutation(perm_random)

    initial_list = list(range(N))
    random.shuffle(initial_list)
    permuted_list = permute_list(initial_list, perm_random)
    permuted_back = permute_list(permuted_list, inv_perm)
    assert initial_list == permuted_back


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
