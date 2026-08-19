"""
Before reading this code, it is important to understand the two levels at which
torch operations can be asynchronous, at the GPU level, and at the python level.

The GPU is a massively parallel device, and it is possible to submit many independent
compute kernels to it, where it can work on them simultaneously if hardware bandwidth
is available. Signifying work dependencies happens through CUDA streams. Work on the same
stream is done sequentially, where the next operation is only started after the previous
one is finished, even if there is no actual data dependency between the operations
(such as when copying tensor A to GPU, and then doing a computation on tensor B).
Submitting the operations on different streams will make them happen in parallel if
hardware bandwidth is available (for the example of copy and compute this will be true
since copy and compute use different modules in the hardware.). However, in that case,
if a result from stream 1 is needed by stream 2, one must explicitly issue a wait
instruction on stream 2 to force it to wait until the operation in stream 1 is done,
otherwise data corruption or crashes will occur (see torch.cuda.Stream.wait_stream).
By default, torch puts everything on the default stream, unless a with statement is used.

Secondly, torch is mostly asynchronous at the python level. Some function calls
(mostly copies to/from GPU) only return upon completion, but ones such as tensordot
return when the kernel is submitted to GPU even if the computation is not done.
This means that the python code will submit compute jobs the GPU as fast as possible
to ensure maximum parallelization on the GPU until it hits a print or something,
where it will wait for the work to complete to copy it to CPU and show the results.

It is possible to copy data between CPU and GPU completely asynchronously by
putting the copy on a different stream (for the GPU), calling the copy with
non_blocking=True (for the python) and making sure the CPU memory is page-locked
using pin_memory=True. The latter prevents the operating system from doing
virtual memory shenanigans, and is a requirement for the GPU to do asynchronous
memory transfers at the hardware level.

Bath tensors are only used two at a time (see MPSBackendImpl._evolve), but a whole
stack of them is kept alive during the TDVP/DMRG sweeps. The helpers below move
tensors to pinned CPU memory, and fetch them back to the GPU.
All copies are asynchronous, on a dedicated CUDA stream, so that
they can overlap with computations on the compute (default) stream.
The user is responsible to call wait_for_transfers and synchronize_transfers
to ensure synchronization between the transfer stream and the default stream.
"""

from typing import Optional
import torch

_transfer_stream: Optional[torch.cuda.Stream] = None  # lazy: keep CUDA uninitialized
# We need to keep track of active GPU->CPU transfers,
# so we will know when the GPU data can be freed
_pending_frees: list[tuple[torch.cuda.Event, torch.Tensor]] = []
_MAX_PENDING_FREES = 1


def _get_transfer_stream() -> torch.cuda.Stream:
    global _transfer_stream
    if _transfer_stream is None:
        _transfer_stream = torch.cuda.Stream()
    return _transfer_stream


def offload_bath_to_cpu(
    bath: torch.Tensor, max_pending_frees: int | None = None
) -> torch.Tensor:
    """
    Start an asynchronous copy of the given bath tensor to pinned CPU memory,
    returning the CPU tensor. Store the GPU tensor, together with an event signifying
    the copy is done, so we know when the GPU tensor can be freed. Checking and cleanup
    happens in this function and in wait_for_transfers and synchronize_transfers.
    """
    max_pending_frees = max_pending_frees or _MAX_PENDING_FREES
    if not bath.is_cuda:
        return bath
    result = torch.empty(bath.shape, dtype=bath.dtype, device="cpu", pin_memory=True)
    stream = _get_transfer_stream()
    stream.wait_stream(torch.cuda.current_stream())  # bath must be fully computed
    with torch.cuda.stream(stream):
        result.copy_(bath, non_blocking=True)
    event = torch.cuda.Event()
    event.record(stream)
    # Copies complete in FIFO order on the transfer stream: release the tensors
    # whose copy is done, and block on the oldest copy if too many are in flight.
    while _pending_frees and (
        _pending_frees[0][0].query() or len(_pending_frees) > max_pending_frees
    ):
        _pending_frees[0][0].synchronize()  # no-op if the copy is already done
        _pending_frees.pop(0)
    _pending_frees.append((event, bath))
    return result


def fetch_bath_from_cpu(bath: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Start an asynchronous copy of the given bath tensor to the given device.
    The caller must call wait_for_transfers() between this call and the first
    use (or deallocation) of the result on the compute stream.
    """
    if bath.device.type == device.type:
        return bath
    # Allocated on the compute stream, so that its later use and deallocation
    # there are stream-ordered.
    result = torch.empty_like(bath, device=device)
    stream = _get_transfer_stream()
    # The destination memory may be recycled from earlier compute stream work;
    # the copy must be ordered after that.
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result.copy_(bath, non_blocking=True)
    return result


def wait_for_transfers() -> None:
    """
    Order all subsequent GPU computations after the pending bath transfers.
    This does not block the host.
    """
    if _transfer_stream is not None:
        torch.cuda.current_stream().wait_stream(_transfer_stream)
    # Any deallocation now happens after the compute stream was ordered
    # after the copies reading these tensors.
    _pending_frees.clear()


def synchronize_transfers() -> None:
    """
    Make the host wait for all pending bath transfers, so that offloaded
    tensors can safely be read on the CPU (e.g. for pickling).
    """
    if _transfer_stream is not None:
        _transfer_stream.synchronize()
    _pending_frees.clear()
