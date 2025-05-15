import torch
import pytest

from emu_base.utils import deallocate_tensor


def test_deallocate_tensor_shape():
    t = torch.ones(100, 100, dtype=torch.complex128)

    deallocate_tensor(t)

    assert len(t.shape) == 1
    assert t.shape[0] == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test needs a GPU with CUDA installed"
)
def test_deallocate_tensor_free_gpu_memory():
    original_memory_allocated = torch.cuda.memory_allocated()

    t = torch.ones(100, 100, dtype=torch.complex128, device="cuda:0")

    assert torch.cuda.memory_allocated() == original_memory_allocated + 160256

    def do_stuff(a_tensor):
        deallocate_tensor(a_tensor)

    do_stuff(t)

    assert torch.cuda.memory_allocated() == original_memory_allocated

    # t's memory is gone even though the object still exists for Python
    # (as referenced by the `t` name)
