from emu_ct.pulser_adapter import _extract_omega_delta
import torch
from unittest.mock import patch, MagicMock
import pytest

TEST_DURATION = 31
TEST_QUBIT_IDS = ["test_qubit_0", "test_qubit_1", "test_qubit_2"]


def make_channel_samples_mock(channels_data):
    channel_samples = MagicMock()
    channel_samples.amp = torch.zeros(TEST_DURATION)
    channel_samples.det = torch.zeros(TEST_DURATION)
    channel_samples.slots = []
    for slot_data, slot_samples in channels_data.items():
        slot = MagicMock()
        slot.ti = slot_data[0]
        slot.tf = slot_data[1]
        assert slot_data[0] < slot_data[1]
        assert len(slot_samples) == slot.tf - slot.ti
        channel_samples.amp[slot.ti : slot.tf] = torch.Tensor(
            [x[0] for x in slot_samples]
        )
        channel_samples.det[slot.ti : slot.tf] = torch.Tensor(
            [x[1] for x in slot_samples]
        )
        slot.targets = slot_data[2]
        channel_samples.slots.append(slot)

    channel_samples.amp.requires_grad = True
    channel_samples.det.requires_grad = True

    return channel_samples


def make_sequence_samples_mock(*channels):
    """
    Creates a mock samples sequence, specified by channels containing slots and samples.
    Each channel should be a dictionary
        { (ti, tf, (qubits, ...)): [(amp0, det0), (amp1, det1), ...], ...}
    where
        ti: time at beginning of slot
        tf: time at end of slot (tf is not included in the slot)
        qubits: qubit id strings
        amp0, amp1, ...: amplitude samples for that slot
        det0, det1, ...: detuning samples for that slot


    Slots don't have to be consecutive.
    """
    sequence_samples = MagicMock()
    sequence_samples.samples_list = [make_channel_samples_mock(c) for c in channels]
    sequence_samples.max_duration = TEST_DURATION

    return sequence_samples


sequence = MagicMock()
sequence.register.qubit_ids = TEST_QUBIT_IDS


@patch("emu_ct.pulser_adapter.pulser.sampler.sampler.sample")
def test_single_channel(mock_pulser_sample):
    mock_pulser_sample.return_value = make_sequence_samples_mock(
        {
            (1, 2, ("test_qubit_1",)): [(-2, -1)],
            (10, 20, ("test_qubit_0",)): [(1, 2)] * 10,
            (26, 30, ("test_qubit_2",)): [(3, 4), (5, 6), (7, 8), (9, 10)],
        }
    )

    dt = 2
    actual_omega, actual_delta = _extract_omega_delta(sequence, dt, False)

    expected_number_of_samples = 15  # ceil(TEST_DURATION / sampling_rate -0.5)
    expected_omega = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_delta = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_omega[0, 1] = -2
    expected_delta[0, 1] = -1
    expected_omega[5:10, 0] = 1
    expected_delta[5:10, 0] = 2
    expected_omega[13, 2] = 5
    expected_delta[13, 2] = 6
    expected_omega[14, 2] = 9
    expected_delta[14, 2] = 10

    assert torch.allclose(actual_omega, expected_omega)
    assert torch.allclose(actual_delta, expected_delta)


# @patch("emu_ct.pulser_adapter.pulser.sampler.sampler.sample")
# def test_autograd(mock_pulser_sample):
#    mock_pulser_sample.return_value = make_sequence_samples_mock(
#        {
#            (10, 20, ("test_qubit_0",)): [(1, 2)] * 10,
#            (26, 29, ("test_qubit_2",)): [(3, 4), (5, 6), (7, 8)],
#        }
#    )
#
#    dt = 2
#    res = torch.autograd.grad(
#        _extract_omega_delta(sequence, dt, False)[0][13, 2].real,
#        mock_pulser_sample.return_value.samples_list[0].amp,
#    )
#    expected = torch.zeros(30)
#    expected[26] = 1
#    assert torch.allclose(res[0], expected)


@patch("emu_ct.pulser_adapter.pulser.sampler.sampler.sample")
def test_multiple_channels(mock_pulser_sample):
    mock_pulser_sample.return_value = make_sequence_samples_mock(
        {
            (10, 20, ("test_qubit_0",)): [(1, 2)] * 10,
        },
        {
            (
                26,
                30,
                (
                    "test_qubit_0",
                    "test_qubit_2",
                ),
            ): [(3, 4), (5, 6), (7, 8), (9, 10)],
        },
    )
    dt = 2
    actual_omega, actual_delta = _extract_omega_delta(sequence, dt, False)
    expected_number_of_samples = 15  # TEST_DURATION // sampling_rate + 1
    expected_omega = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_delta = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_omega[5:10, 0] = 1
    expected_delta[5:10, 0] = 2
    expected_omega[13, (0, 2)] = 5
    expected_delta[13, (0, 2)] = 6
    expected_omega[14, (0, 2)] = 9
    expected_delta[14, (0, 2)] = 10
    assert torch.allclose(actual_omega, expected_omega)
    assert torch.allclose(actual_delta, expected_delta)


@patch("emu_ct.pulser_adapter.pulser.sampler.sampler.sample")
def test_multiple_channels_together(mock_pulser_sample):
    mock_pulser_sample.return_value = make_sequence_samples_mock(
        {
            (10, 20, ("test_qubit_0",)): [(1, 2)] * 10,
        },
        {
            (10, 20, ("test_qubit_1", "test_qubit_2")): [(3, 4)] * 10,
        },
    )
    dt = 5
    actual_omega, actual_delta = _extract_omega_delta(sequence, dt, False)
    expected_number_of_samples = 6
    expected_omega = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_delta = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_omega[2:4, 0] = 1
    expected_delta[2:4, 0] = 2
    expected_omega[2:4, 1:3] = 3
    expected_delta[2:4, 1:3] = 4
    assert torch.allclose(actual_omega, expected_omega)
    assert torch.allclose(actual_delta, expected_delta)


@patch("emu_ct.pulser_adapter.pulser.sampler.sampler.sample")
def test_multiple_channels_together_on_same_qubit(mock_pulser_sample):
    mock_pulser_sample.return_value = make_sequence_samples_mock(
        {
            (10, 20, ("test_qubit_0",)): [(1, 2)] * 10,
        },
        {
            (10, 20, ("test_qubit_0", "test_qubit_1")): [(1, 2)] * 10,
        },
    )
    with pytest.raises(NotImplementedError) as exception_info:
        dt = 2
        _extract_omega_delta(sequence, dt, False)
    assert "multiple pulses acting on same qubit" in str(exception_info.value)
