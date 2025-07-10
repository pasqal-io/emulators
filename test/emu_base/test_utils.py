from collections import Counter
import random
from unittest.mock import call, patch
import torch
import pytest
from emu_base.utils import deallocate_tensor
from emu_base.utils import readout_with_error, apply_measurement_errors


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


@patch("emu_base.utils.random.random")
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
    with patch("emu_base.utils.readout_with_error") as readout_with_error_mock:
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

        expected_calls = [
            call("1", **ps),
            call("0", **ps),
            call("1", **ps),
            call("0", **ps),
            call("1", **ps),
            call("0", **ps),
            call("1", **ps),
            call("0", **ps),
            call("1", **ps),
            call("0", **ps),
            call("1", **ps),
            call("0", **ps),
            call("0", **ps),
            call("1", **ps),
            call("0", **ps),
            call("1", **ps),
            call("0", **ps),
            call("1", **ps),
            call("0", **ps),
            call("1", **ps),
            call("1", **ps),
            call("1", **ps),
            call("1", **ps),
            call("1", **ps),
        ]

        readout_with_error_mock.assert_has_calls(expected_calls)

        assert readout_with_error_mock.call_count == len(expected_calls)

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
