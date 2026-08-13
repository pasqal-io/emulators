import math

import pytest
import torch

from emu_base import DEVICE_COUNT
from emu_mps.utils import (
    extended_mpo_factors,
    extended_mps_factors,
    fetch_bath_from_cpu,
    offload_bath_to_cpu,
    split_matrix,
    get_extended_site_index,
    tensor_trace,
    wait_for_transfers,
)


@pytest.mark.parametrize(
    "orth_center_right",
    [
        (True),
        (False),
    ],
)
def test_split_matrix(orth_center_right):
    a = torch.diag(torch.tensor([1.0, 5.0, 3.0, 6.0, -2.0]))

    l, r = split_matrix(
        a, max_rank=3, max_error=9999, orth_center_right=orth_center_right
    )

    m = l.T.conj() @ l if orth_center_right else r @ r.T.conj()
    assert torch.allclose(m, torch.eye(3, 3))
    assert torch.allclose(l @ r, torch.diag(torch.tensor([0.0, 5.0, 3.0, 6.0, 0.0])))

    l, r = split_matrix(
        a, max_rank=4, max_error=9999, orth_center_right=orth_center_right
    )

    m = l.T.conj() @ l if orth_center_right else r @ r.T.conj()
    assert torch.allclose(m, torch.eye(4, 4))
    assert torch.allclose(l @ r, torch.diag(torch.tensor([0.0, 5.0, 3.0, 6.0, -2.0])))

    l, r = split_matrix(
        a, max_rank=20, max_error=1.5, orth_center_right=orth_center_right
    )

    m = l.T.conj() @ l if orth_center_right else r @ r.T.conj()
    assert torch.allclose(m, torch.eye(4, 4))
    assert torch.allclose(l @ r, torch.diag(torch.tensor([0.0, 5.0, 3.0, 6.0, -2.0])))

    l, r = split_matrix(
        a, max_rank=20, max_error=math.sqrt(5) + 0.1, orth_center_right=orth_center_right
    )

    m = l.T.conj() @ l if orth_center_right else r @ r.T.conj()
    assert torch.allclose(m, torch.eye(3, 3))
    assert torch.allclose(l @ r, torch.diag(torch.tensor([0.0, 5.0, 3.0, 6.0, 0.0])))


def test_split_matrix_norm():
    a = torch.randn(5, 5)
    n = torch.linalg.norm(a)
    q1, r1 = split_matrix(a, max_rank=2)
    n1 = torch.linalg.norm(r1)
    q2, r2 = split_matrix(a, max_rank=2, preserve_norm=True)
    n2 = torch.linalg.norm(r2)
    assert torch.allclose(q1, q2)
    assert torch.allclose(n2, n)
    assert torch.allclose(r2, (n2 / n1) * r1)
    assert torch.allclose(q2 @ r2, (n2 / n1) * q1 @ r1)

    l1, q1 = split_matrix(a, max_rank=2, orth_center_right=False)
    n1 = torch.linalg.norm(l1)
    l2, q2 = split_matrix(a, max_rank=2, orth_center_right=False, preserve_norm=True)
    n2 = torch.linalg.norm(l2)
    assert torch.allclose(q1, q2)
    assert torch.allclose(n2, n)
    assert torch.allclose(l2, (n2 / n1) * l1)
    assert torch.allclose(l2 @ q2, (n2 / n1) * l1 @ q1)


def test_bath_transfers_cpu_noop():
    bath = torch.rand(3, 5, 3, dtype=torch.complex128)

    assert offload_bath_to_cpu(bath) is bath
    assert fetch_bath_from_cpu(bath, torch.device("cpu")) is bath
    wait_for_transfers()  # no-op without pending transfers


@pytest.mark.skipif(DEVICE_COUNT == 0, reason="Requires a GPU")
def test_bath_transfers_gpu_roundtrip():
    bath = torch.rand(3, 5, 3, dtype=torch.complex128, device="cuda")

    offloaded = offload_bath_to_cpu(bath)
    assert not offloaded.is_cuda
    assert offload_bath_to_cpu(offloaded) is offloaded

    fetched = fetch_bath_from_cpu(offloaded, bath.device)
    assert fetched.is_cuda
    assert fetch_bath_from_cpu(fetched, bath.device) is fetched

    wait_for_transfers()
    assert torch.equal(fetched, bath)


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


def test_get_extended_site_index():
    T, F = True, False
    assert get_extended_site_index([T, F, F, T, T, F, T, F], None) is None
    assert get_extended_site_index([T, F, F, T, T, F, T, F], 0) == 0
    assert get_extended_site_index([T, F, F, T, T, F, T, F], 1) == 3
    assert get_extended_site_index([T, F, F, T, T, F, T, F], 2) == 4
    assert get_extended_site_index([T, F, F, T, T, F, T, F], 3) == 6

    with pytest.raises(ValueError) as e:
        get_extended_site_index([T, F, F, T, T, F, T, F], 4)
    assert str(e.value) == "Index 4 does not exist"


def test_tensor_trace():
    t = torch.rand(3, 3, 3)

    contracted_01 = tensor_trace(t, 0, 1)
    assert torch.allclose(contracted_01, tensor_trace(t, 1, 0))
    assert contracted_01.shape == (3,)
    assert contracted_01[0] == pytest.approx(t[0, 0, 0] + t[1, 1, 0] + t[2, 2, 0])
    assert contracted_01[1] == pytest.approx(t[0, 0, 1] + t[1, 1, 1] + t[2, 2, 1])

    contracted_12 = tensor_trace(t, 1, 2)
    assert torch.allclose(contracted_12, tensor_trace(t, 2, 1))
    assert contracted_12.shape == (3,)
    assert contracted_12[0] == pytest.approx(t[0, 0, 0] + t[0, 1, 1] + t[0, 2, 2])
    assert contracted_12[1] == pytest.approx(t[1, 0, 0] + t[1, 1, 1] + t[1, 2, 2])

    with pytest.raises(AssertionError) as e:
        tensor_trace(torch.rand(2, 3, 4), 1, 2)

    assert str(e.value) == "dimensions should match"
