from emu_base import BackendConfig
import pytest
import torch
import numpy


def test_interaction_matrix_type() -> None:
    BackendConfig(interaction_matrix=None)  # None is ok, empty tensor not ok

    def test_input_type(int_mat) -> None:
        with pytest.raises(
            ValueError,
            match="Interaction matrix must be provided as a Python list of lists of floats",
        ):
            BackendConfig(interaction_matrix=int_mat)

    test_input_type(int_mat=[[0, 1], [1, 0]])  # not float
    test_input_type(int_mat=[0, 1, 2, 3, 4])  # not list of lists
    test_input_type(int_mat=torch.eye(5))  # not list of lists
    test_input_type(int_mat=numpy.eye(5))  # not list of lists


@pytest.mark.parametrize(
    "n_qubits",
    [10, 15, 20],
)
def test_interaction_matrix_properties(n_qubits: int) -> None:
    def test_symmetric_zero_diag(int_mat: torch.tensor) -> None:
        with pytest.raises(
            ValueError, match="Interaction matrix is not symmetric and zero diag"
        ):
            BackendConfig(interaction_matrix=int_mat.tolist())

    wrong_interaction_matrix = torch.randn(
        n_qubits, n_qubits + 1, dtype=torch.float64
    )  # not square
    test_symmetric_zero_diag(wrong_interaction_matrix)

    wrong_interaction_matrix = torch.randn(
        n_qubits, n_qubits - 1, dtype=torch.float64
    )  # not square
    test_symmetric_zero_diag(wrong_interaction_matrix)

    wrong_interaction_matrix = torch.randn(
        n_qubits, n_qubits, dtype=torch.float64
    )  # not symmetric
    test_symmetric_zero_diag(wrong_interaction_matrix)

    wrong_interaction_matrix = (
        wrong_interaction_matrix + wrong_interaction_matrix.T
    )  # diagonal not 0
    test_symmetric_zero_diag(wrong_interaction_matrix)

    correct_interaction_matrix = wrong_interaction_matrix.fill_diagonal_(0).tolist()
    BackendConfig(
        interaction_matrix=correct_interaction_matrix,
    )
