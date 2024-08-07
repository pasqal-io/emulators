import pytest
import torch
import math
from typing import List
from emu_ct.utils import (
    split_tensor,
    assign_devices,
    extended_mps_factors,
    extended_mpo_factors,
    apply_measurement_errors,
    readout_with_error,
)
from collections import Counter
import random
from unittest.mock import patch, call


@patch("emu_ct.noise.random.random")
def test_readout_with_error(random_mock):
    random_mock.side_effect = [0.6, 0.08, 0.4, 0.1, 0.04]

    assert readout_with_error("0", p_false_pos=0.1, p_false_neg=0.2) == "0"
    assert readout_with_error("0", p_false_pos=0.1, p_false_neg=0.2) == "1"
    assert readout_with_error("1", p_false_pos=0.1, p_false_neg=0.2) == "1"
    assert readout_with_error("1", p_false_pos=0.1, p_false_neg=0.2) == "0"
    assert readout_with_error("0", p_false_pos=0.1, p_false_neg=0.2) == "1"


def test_add_measurement_errors():
    bitstrings = Counter(
        {
            "1010": 3,
            "0101": 2,
            "1111": 1,
        }
    )

    # Error probability is null
    bitstrings_without_measurement_errors = apply_measurement_errors(
        bitstrings, p_false_pos=0.0, p_false_neg=0.0
    )
    assert bitstrings_without_measurement_errors == bitstrings

    # Error probability is not null
    p_false_pos = 0.1
    p_false_neg = 0.2
    with patch("emu_ct.utils.readout_with_error") as readout_with_error_mock:
        readout_with_error_mock.side_effect = [
            "1",
            "0",
            "1",
            "1",
            "0",
            "0",
            "0",
            "1",
            "1",
            "0",
            "1",
            "0",
            "0",
            "1",
            "0",
            "1",
            "0",
            "0",
            "0",
            "1",
            "1",
            "1",
            "1",
            "1",
        ]

        bitstrings_with_measurement_errors = apply_measurement_errors(
            bitstrings, p_false_pos=p_false_pos, p_false_neg=p_false_neg
        )

        ps = {"p_false_pos": p_false_pos, "p_false_neg": p_false_neg}

        # Counter is a subclass of dict.
        # In Python 3.7+, dict iteration order is guaranteed to be the insertion order.
        # Therefore, the loop in add_measurement_errors has a known order.

        readout_with_error_mock.assert_has_calls(
            [call("1", **ps), call("0", **ps), call("1", **ps), call("0", **ps)] * 3
            + [call("0", **ps), call("1", **ps), call("0", **ps), call("1", **ps)] * 2
            + [call("1", **ps), call("1", **ps), call("1", **ps), call("1", **ps)] * 1
        )

        assert bitstrings_with_measurement_errors == Counter(
            {"1011": 1, "0001": 2, "1010": 1, "0101": 1, "1111": 1}
        )


def test_add_measurement_errors_large():
    random.seed(0xDEADBEEF)

    bitstrings = Counter(
        {
            "101010001": 39845,
            "010110001": 2,
            "111100001": 1,
        }
    )

    bitstrings_with_measurement_errors = apply_measurement_errors(
        bitstrings, p_false_pos=0.000001, p_false_neg=0.0000085
    )
    assert bitstrings_with_measurement_errors == Counter(
        {
            "101010001": 39843,
            "101010000": 1,
            "111010001": 1,
            "010110001": 2,
            "111100001": 1,
        }
    )

    assert sum(bitstrings_with_measurement_errors.values()) == sum(bitstrings.values())


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
