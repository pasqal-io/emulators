import pytest
import torch
from collections import Counter
from unittest.mock import patch
from emu_sv import StateVector, DensityMatrix
from emu_sv import dense_state


def test_inner_and_overlap() -> None:
    factor = 1.0j
    dtype = torch.complex128
    tensor1 = torch.tensor([factor, 0, 0, 0, 0, 0, 0, factor], dtype=dtype)
    tensor2 = torch.tensor([0, factor, 0, 0, 0, 0, 0, factor], dtype=dtype)

    state1 = StateVector(tensor1)
    tmp = StateVector(tensor2)
    state2 = DensityMatrix.from_state_vector(tmp)

    overlap1 = state1.overlap(state2)  # testing overlap
    overlap2 = state2.overlap(state1)
    expected = torch.tensor([1.0], dtype=dtype, device=state1.data.device)

    assert torch.allclose(overlap1, expected)
    assert torch.allclose(overlap2, expected)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_sample_goes_through_sample_indices(density_matrix: bool) -> None:
    amplitudes = torch.tensor([0.6, 0, 0, 0, 0, 0, 0, 0.8j], dtype=torch.complex128)
    state = StateVector(amplitudes, gpu=False)
    if density_matrix:
        state = DensityMatrix.from_state_vector(state)

    with patch.object(
        dense_state, "sample_indices", autospec=True, return_value=torch.tensor([7, 0, 7])
    ) as sample_indices:
        counts = state.sample(num_shots=3)

    assert counts == Counter({"111": 2, "000": 1})
    sample_indices.assert_called_once()
    probabilities, num_shots = sample_indices.call_args.args
    assert num_shots == 3
    # |0.6|**2 and |0.8j|**2 in float64: a few ulp of rounding
    expected = torch.zeros(8, dtype=torch.float64)
    expected[0], expected[7] = 0.36, 0.64
    torch.testing.assert_close(probabilities, expected, rtol=0, atol=1e-15)
