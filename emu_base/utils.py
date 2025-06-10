import torch
import os

unix_like = os.name != "nt"
if unix_like:
    from resource import RUSAGE_SELF, getrusage


def deallocate_tensor(t: torch.Tensor) -> None:
    """
    Free the memory used by a tensor. This is done regardless of the
    memory management done by Python: it is a forced deallocation
    that ignores the current reference count of the Tensor object.

    It is useful when you want to free memory that is no longer used
    inside a function but that memory is also owned by a variable
    in the outer scope, making it impossible to free it otherwise.

    After calling that function, the Tensor object
    should no longer be used.

    To work properly with e.g. tensordot but also user-created views,
    and since every view of a tensor owns the tensor's storage independently,
    it has to change the storage of the base AND every view referring to the base.
    However, it is not possible to access the views from the base, so
    if there are extra inaccessible views, it will raise an exception.
    """
    if (t._base is None and t._use_count() > 1) or (  # type: ignore[attr-defined]
        t._base is not None and t._base._use_count() > 2  # type: ignore[attr-defined]
    ):
        raise RuntimeError("Cannot deallocate tensor")

    replacement_storage = torch.zeros(0, dtype=t.dtype, device=t.device).untyped_storage()

    t.resize_(0)
    t.set_(source=replacement_storage)

    if t._base is not None:
        t._base.resize_(0)
        t._base.set_(source=replacement_storage)


def get_max_rss(gpu: bool) -> float:
    if gpu:
        max_mem_per_device = (
            torch.cuda.max_memory_allocated(device) * 1e-6
            for device in range(torch.cuda.device_count())
        )
        max_mem = max(max_mem_per_device)
    elif unix_like:
        max_mem = getrusage(RUSAGE_SELF).ru_maxrss * 1e-3
    else:
        return 0.0
    return max_mem
