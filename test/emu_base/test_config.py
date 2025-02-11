import pytest
import torch

from emu_base.base_classes.config import BackendConfig


@pytest.mark.parametrize(
    "n_qubits",
    [10, 25, 50],
)
def test_interaction_matrix_shape(n_qubits):
    def test_assert_make_H(int_mat: int) -> None:
        with pytest.raises(
            AssertionError, match="Interaction matrix is not symmetric and zero diag"
        ):
            BackendConfig(interaction_matrix=int_mat.tolist())

    wrong_interaction_matrix = torch.tensor([[]])  # empty matrix
    test_assert_make_H(wrong_interaction_matrix)

    wrong_interaction_matrix = torch.randn(
        n_qubits, n_qubits, n_qubits, dtype=torch.float64
    )  # 3D not matrix
    test_assert_make_H(wrong_interaction_matrix)

    wrong_interaction_matrix = torch.randn(
        n_qubits, n_qubits + 1, dtype=torch.float64
    )  # not square
    test_assert_make_H(wrong_interaction_matrix)

    wrong_interaction_matrix = torch.randn(
        n_qubits, n_qubits - 1, dtype=torch.float64
    )  # not square
    test_assert_make_H(wrong_interaction_matrix)

    wrong_interaction_matrix = torch.randn(
        n_qubits, n_qubits, dtype=torch.float64
    )  # not symmetric
    test_assert_make_H(wrong_interaction_matrix)

    wrong_interaction_matrix = (
        wrong_interaction_matrix + wrong_interaction_matrix.T
    )  # diagonal not 0
    test_assert_make_H(wrong_interaction_matrix)

    correct_interaction_matrix = wrong_interaction_matrix.fill_diagonal_(0)
    correct_interaction_matrix = correct_interaction_matrix.tolist()
    BackendConfig(
        interaction_matrix=correct_interaction_matrix,
    )
