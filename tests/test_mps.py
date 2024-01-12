from emu_ct import MPS, Config
import torch


def test_init():
    Config().set_num_devices_to_use(0)
    factor1 = torch.tensor([[[0, 1, 0, 0], [0, 0, 0, 0]]], dtype=torch.complex128)
    factor2 = torch.tensor(
        [
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=torch.complex128,
    )
    factor3 = torch.tensor(
        [[[0], [0]], [[0], [0]], [[0], [0]], [[1], [0]], [[0], [0]]],
        dtype=torch.complex128,
    )
    state = MPS([factor1, factor2, factor3], truncate=True)
    for factor in state.factors:
        assert torch.allclose(factor, torch.tensor([[[1], [0]]], dtype=torch.complex128))
