import torch
import math
from typing import List
from emu_ct.utils import truncated_svd, assign_devices


def test_truncated_svd():
    a = torch.diag(torch.tensor([1.0, 5.0, 3.0, 6.0, -2.0]))

    u, d, vh = truncated_svd(a, max_rank=3, max_error=9999)

    assert torch.allclose(d, torch.tensor([6.0, 5.0, 3.0]))

    u, d, vh = truncated_svd(a, max_rank=4, max_error=9999)

    assert torch.allclose(d, torch.tensor([6.0, 5.0, 3.0, 2.0]))

    u, d, vh = truncated_svd(a, max_rank=20, max_error=1.5)

    assert torch.allclose(d, torch.tensor([6.0, 5.0, 3.0, 2.0]))

    u, d, vh = truncated_svd(a, max_rank=20, max_error=math.sqrt(5) + 0.1)

    assert torch.allclose(d, torch.tensor([6.0, 5.0, 3.0]))


def test_assign_devices():
    class MockTensor:
        def __init__(self):
            self.device = "unset"

        def to(self, s: str):
            copy = MockTensor()
            copy.device = s
            return copy

    def gpus(mock_tensors: List[MockTensor]) -> List[int]:
        result = []
        for mock_tensor in mock_tensors:
            assert mock_tensor.device[:4] == "cuda"
            assert len(mock_tensor.device) == 6
            result.append(int(mock_tensor.device[5]))
        return result

    ts = [MockTensor() for _ in range(13)]

    assert ts[0].device == "unset"

    assign_devices(ts, num_devices_to_use=0)

    assert ts[0].device == "cpu"

    assign_devices(ts, num_devices_to_use=1)

    assert gpus(ts) == [0] * 13

    assign_devices(ts, num_devices_to_use=3)

    assert gpus(ts) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]

    assign_devices(ts, num_devices_to_use=5)

    assert gpus(ts) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]

    assign_devices(ts, num_devices_to_use=2)

    assert gpus(ts) == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
