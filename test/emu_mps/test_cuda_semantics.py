import torch
import pytest

from emu_base import DEVICE_COUNT
import emu_mps

from emu_mps.cuda_semantics import (
    fetch_bath_from_cpu,
    offload_bath_to_cpu,
    wait_for_transfers,
)

dtype = torch.complex128


def test_bath_transfers_cpu_noop():
    bath = torch.rand(3, 5, 3, dtype=dtype)

    assert offload_bath_to_cpu(bath) is bath
    assert fetch_bath_from_cpu(bath, torch.device("cpu")) is bath
    wait_for_transfers()  # no-op without pending transfers


@pytest.mark.skipif(DEVICE_COUNT == 0, reason="Requires a GPU")
def test_bath_transfers_gpu_roundtrip():
    bath = torch.rand(3, 5, 3, dtype=dtype, device="cuda")

    offloaded = offload_bath_to_cpu(bath)
    assert not offloaded.is_cuda
    assert offload_bath_to_cpu(offloaded) is offloaded
    # The GPU source is kept alive until its copy is done.
    assert any(t is bath for _, t in emu_mps.cuda_semantics._pending_frees)

    fetched = fetch_bath_from_cpu(offloaded, bath.device)
    assert fetched.is_cuda
    assert fetch_bath_from_cpu(fetched, bath.device) is fetched

    wait_for_transfers()
    assert emu_mps.cuda_semantics._pending_frees == []
    assert torch.equal(fetched, bath)


@pytest.mark.skipif(DEVICE_COUNT == 0, reason="Requires a GPU")
def test_offloaded_bath_survives_memory_reuse():
    # The GPU source of an in-flight offload must not have its memory reused
    # by subsequent compute stream allocations before the copy is done.
    reference = torch.rand(256, 20, 256, dtype=dtype)
    bath = reference.to("cuda")
    torch.cuda.synchronize()

    offloaded = offload_bath_to_cpu(bath)
    del bath  # drop the caller reference; only _pending_frees keeps it alive

    # Allocation-heavy work on the compute stream, trying to reuse the memory.
    for _ in range(10):
        junk = torch.zeros(256, 20, 256, dtype=dtype, device="cuda")
        del junk

    wait_for_transfers()
    torch.cuda.synchronize()
    assert torch.equal(offloaded, reference)


@pytest.mark.skipif(DEVICE_COUNT == 0, reason="Requires a GPU")
def test_offload_pending_frees_bounded():
    # Offloading many baths back-to-back (as in init_baths) must not retain
    # more than _MAX_PENDING_FREES GPU tensors awaiting their copy.
    wait_for_transfers()
    baths = [torch.rand(64, 4, 64, dtype=dtype, device="cuda") for _ in range(8)]
    offloaded = [offload_bath_to_cpu(bath) for bath in baths]
    assert (
        len(emu_mps.cuda_semantics._pending_frees)
        <= emu_mps.cuda_semantics._MAX_PENDING_FREES + 1
    )

    wait_for_transfers()
    torch.cuda.synchronize()
    assert all(torch.equal(off, bath.cpu()) for off, bath in zip(offloaded, baths))
