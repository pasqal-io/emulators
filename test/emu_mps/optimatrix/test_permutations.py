import pytest
import torch

from emu_mps.optimatrix.permutations import (
    permute_string,
    eye_permutation,
    inv_permutation,
    permute_tensor,
    permute_list,
)


def test_eye_permutation() -> None:
    assert torch.equal(eye_permutation(5), torch.tensor([0, 1, 2, 3, 4]))


def test_permute_list() -> None:
    perm = torch.tensor([1, 0])
    input_list = ["a", "b"]
    assert permute_list(input_list, perm) == ["b", "a"]

    # 6 permutations of [a, b, c]
    input_list = ["a", "b", "c"]
    assert permute_list(input_list, torch.tensor([0, 1, 2])) == ["a", "b", "c"]
    assert permute_list(input_list, torch.tensor([0, 2, 1])) == ["a", "c", "b"]
    assert permute_list(input_list, torch.tensor([1, 0, 2])) == ["b", "a", "c"]
    assert permute_list(input_list, torch.tensor([1, 2, 0])) == ["b", "c", "a"]
    assert permute_list(input_list, torch.tensor([2, 0, 1])) == ["c", "a", "b"]
    assert permute_list(input_list, torch.tensor([2, 1, 0])) == ["c", "b", "a"]


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


def test_implementation_permute_tensor_1D() -> None:
    vector = torch.tensor([10, 20, 30])
    perm = torch.tensor([2, 0, 1])

    expected = torch.tensor([30, 10, 20])
    assert torch.all(permute_tensor(vector, perm) == expected)


def test_implementation_permute_tensor_2D() -> None:
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    perm = torch.tensor([1, 0, 2])

    expected = torch.tensor([[5, 4, 6], [2, 1, 3], [8, 7, 9]])
    assert torch.all(permute_tensor(matrix, perm) == expected)


def test_implementation_permute_tensor_ND() -> None:
    tensor_3d = torch.tensor([[[42]]])
    err_msg = "Only 1D tensors or square 2D tensors are supported."
    with pytest.raises(ValueError, match=err_msg):
        permute_tensor(tensor_3d, torch.tensor([0]))
