from emu_base import BackendConfig
import pytest
import torch
import numpy


def test_interaction_matrix():
    BackendConfig(interaction_matrix=None)  # None is ok, empty tensor not ok
    BackendConfig(interaction_matrix=[[0, 1], [1, 0]])

    def test_BackendConfig(int_mat) -> None:
        with pytest.raises(
            ValueError,
            match="Interaction matrix must be provided as a Python list of lists of floats",
        ):
            BackendConfig(interaction_matrix=int_mat)

    test_BackendConfig(int_mat=[1, 2, 3])
    test_BackendConfig(int_mat=torch.eye(3))
    test_BackendConfig(int_mat=numpy.eye(3))


@pytest.mark.parametrize(
    "n_qubits",
    [10, 25, 50],
)
def test_shape_interaction_matrix(n_qubits):
    def test_BackendConfig(int_mat: int) -> None:
        with pytest.raises(
            ValueError, match="Interaction matrix is not symmetric and zero diag"
        ):
            BackendConfig(interaction_matrix=int_mat.tolist())

    wrong_interaction_matrix = torch.tensor([[]])  # empty matrix
    test_BackendConfig(wrong_interaction_matrix)

    wrong_interaction_matrix = torch.randn(
        n_qubits, n_qubits, n_qubits, dtype=torch.float64
    )  # 3D not matrix
    test_BackendConfig(wrong_interaction_matrix)

    wrong_interaction_matrix = torch.randn(
        n_qubits, n_qubits + 1, dtype=torch.float64
    )  # not square
    test_BackendConfig(wrong_interaction_matrix)

    wrong_interaction_matrix = torch.randn(
        n_qubits, n_qubits - 1, dtype=torch.float64
    )  # not square
    test_BackendConfig(wrong_interaction_matrix)

    wrong_interaction_matrix = torch.randn(
        n_qubits, n_qubits, dtype=torch.float64
    )  # not symmetric
    test_BackendConfig(wrong_interaction_matrix)

    wrong_interaction_matrix = (
        wrong_interaction_matrix + wrong_interaction_matrix.T
    )  # diagonal not 0
    test_BackendConfig(wrong_interaction_matrix)

    correct_interaction_matrix = wrong_interaction_matrix.fill_diagonal_(0)
    correct_interaction_matrix = correct_interaction_matrix.tolist()
    BackendConfig(
        interaction_matrix=correct_interaction_matrix,
    )
