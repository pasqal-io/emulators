import torch
from emu_sv import StateVector, DensityMatrix


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
