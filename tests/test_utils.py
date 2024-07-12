import pytest
import torch
import math
from typing import List
from emu_ct.utils import (
    split_tensor,
    assign_devices,
    extended_mps_factors,
    extended_mpo_factors,
)


@pytest.mark.parametrize(
    "orth_center_right",
    [
        (True),
        (False),
    ],
)
def test_split_tensor(orth_center_right):
    a = torch.diag(torch.tensor([1.0, 5.0, 3.0, 6.0, -2.0]))

    l, r = split_tensor(
        a, max_rank=3, max_error=9999, orth_center_right=orth_center_right
    )

    m = l.T.conj() @ l if orth_center_right else r @ r.T.conj()
    assert torch.allclose(m, torch.eye(3, 3))
    assert torch.allclose(l @ r, torch.diag(torch.tensor([0.0, 5.0, 3.0, 6.0, 0.0])))

    l, r = split_tensor(
        a, max_rank=4, max_error=9999, orth_center_right=orth_center_right
    )

    m = l.T.conj() @ l if orth_center_right else r @ r.T.conj()
    assert torch.allclose(m, torch.eye(4, 4))
    assert torch.allclose(l @ r, torch.diag(torch.tensor([0.0, 5.0, 3.0, 6.0, -2.0])))

    l, r = split_tensor(
        a, max_rank=20, max_error=1.5, orth_center_right=orth_center_right
    )

    m = l.T.conj() @ l if orth_center_right else r @ r.T.conj()
    assert torch.allclose(m, torch.eye(4, 4))
    assert torch.allclose(l @ r, torch.diag(torch.tensor([0.0, 5.0, 3.0, 6.0, -2.0])))

    l, r = split_tensor(
        a, max_rank=20, max_error=math.sqrt(5) + 0.1, orth_center_right=orth_center_right
    )

    m = l.T.conj() @ l if orth_center_right else r @ r.T.conj()
    assert torch.allclose(m, torch.eye(3, 3))
    assert torch.allclose(l @ r, torch.diag(torch.tensor([0.0, 5.0, 3.0, 6.0, 0.0])))


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


def test_extended_mps_factors():
    a = torch.rand(1, 2, 3)
    b = torch.rand(3, 2, 5)
    c = torch.rand(5, 2, 1)
    mpo_factors = [a, b, c]
    where = [False, True, False, True, True, False, False]
    extended = extended_mps_factors(mpo_factors, where)

    assert [t.shape for t in extended] == [
        (1, 2, 1),
        (1, 2, 3),
        (3, 2, 3),
        (3, 2, 5),
        (5, 2, 1),
        (1, 2, 1),
        (1, 2, 1),
    ]
    true_count = 0
    for i, b in enumerate(where):
        if b:
            assert extended[i] is mpo_factors[true_count]
            true_count += 1
        else:
            assert torch.allclose(
                extended[i][:, 0, :],
                torch.eye(extended[i].shape[0], dtype=torch.complex128),
            )
            assert torch.allclose(
                extended[i][:, 1, :],
                torch.zeros(extended[i].shape[0], dtype=torch.complex128),
            )


def test_extended_mpo_factors():
    a = torch.rand(1, 2, 2, 3)
    b = torch.rand(3, 2, 2, 5)
    c = torch.rand(5, 2, 2, 1)
    mpo_factors = [a, b, c]
    where = [False, True, False, True, True, False, False]
    extended = extended_mpo_factors(mpo_factors, where)

    assert [t.shape for t in extended] == [
        (1, 2, 2, 1),
        (1, 2, 2, 3),
        (3, 2, 2, 3),
        (3, 2, 2, 5),
        (5, 2, 2, 1),
        (1, 2, 2, 1),
        (1, 2, 2, 1),
    ]

    true_count = 0
    for i, b in enumerate(where):
        if b:
            assert extended[i] is mpo_factors[true_count]
            true_count += 1
        else:
            assert torch.allclose(
                extended[i][:, 0, 0, :],
                torch.eye(extended[i].shape[0], dtype=torch.complex128),
            )
            assert torch.allclose(
                extended[i][:, 1, 1, :],
                torch.eye(extended[i].shape[0], dtype=torch.complex128),
            )
            assert torch.allclose(
                extended[i][:, 1, 0, :],
                torch.zeros(
                    extended[i].shape[0], extended[i].shape[0], dtype=torch.complex128
                ),
            )
            assert torch.allclose(
                extended[i][:, 0, 1, :],
                torch.zeros(
                    extended[i].shape[0], extended[i].shape[0], dtype=torch.complex128
                ),
            )
