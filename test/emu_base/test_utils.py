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


def test_deallocate_view():
    t = torch.ones(100, 100, dtype=torch.complex128)

    t_view = t.view(-1)

    deallocate_tensor(t_view)

    assert len(t_view.shape) == len(t.shape) == 1
    assert t_view.shape[0] == t.shape[0] == 0


def test_deallocate_base_with_view():
    t = torch.ones(100, 100, dtype=torch.complex128)

    t_view = t.view(-1)

    with pytest.raises(RuntimeError) as e:
        deallocate_tensor(t)

    assert str(e.value) == "Cannot deallocate tensor"

    del t_view

    deallocate_tensor(t)


def test_deallocate_stacked_views():
    t = torch.ones(100, 100, dtype=torch.complex128)

    t_many_view = t.view(-1).view(20, 500).view(2, 5000)

    # The base tensor always has _base == None
    assert t_many_view._base._base is None
    assert t_many_view._base is t

    deallocate_tensor(t_many_view)

    assert len(t_many_view.shape) == len(t.shape) == 1
    assert t_many_view.shape[0] == t.shape[0] == 0


def test_deallocate_multiple_independent_views():
    t = torch.ones(100, 100, dtype=torch.complex128)

    t_view_1 = t.view(-1)

    # The second view prevents deallocation.
    t_view_2 = t.view(-1)

    with pytest.raises(RuntimeError) as e:
        deallocate_tensor(t_view_1)

    assert str(e.value) == "Cannot deallocate tensor"

    del t_view_2

    # Now it works.
    deallocate_tensor(t_view_1)

    assert len(t_view_1.shape) == len(t.shape) == 1
    assert t_view_1.shape[0] == t.shape[0] == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test needs a GPU with CUDA installed"
)
def test_deallocate_tensordot():
    id = torch.eye(100, device="cuda:0")

    original_memory_allocated = torch.cuda.memory_allocated()

    m = torch.tensordot(id, id, dims=1)

    deallocate_tensor(m)

    torch._C._cuda_clearCublasWorkspaces()  # tensordot allocates extra memory on first use

    assert torch.cuda.memory_allocated() == original_memory_allocated
