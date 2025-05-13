import torch


def dist2(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.dist(left, right).item() ** 2


def dist3(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.dist(left, right).item() ** 3


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
    """
    t.resize_(0)
    t.set_(source=torch.zeros(0, dtype=t.dtype, device=t.device).untyped_storage())
