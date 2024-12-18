from emu_base import BackendConfig
import pytest
import torch
import numpy


def test_interaction_matrix():
    BackendConfig(interaction_matrix=None)
    BackendConfig(interaction_matrix=[[1, 1], [1, 1]])

    expected_error = (
        "Interaction matrix must be provided as a Python list of lists of floats"
    )

    with pytest.raises(ValueError) as e:
        BackendConfig(interaction_matrix=[1, 2, 3])

    assert str(e.value) == expected_error

    with pytest.raises(ValueError) as e:
        BackendConfig(interaction_matrix=torch.eye(3))

    assert str(e.value) == expected_error

    with pytest.raises(ValueError) as e2:
        BackendConfig(interaction_matrix=numpy.eye(3))

    assert str(e2.value) == expected_error
