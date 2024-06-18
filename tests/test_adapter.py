from emu_ct.pulser_adapter import extract_omega_delta
import torch
from unittest.mock import patch, MagicMock
import pytest

TEST_DURATION = 30
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
    return sequence_samples


seq = MagicMock()
seq.get_duration.return_value = TEST_DURATION
seq.register.qubit_ids = TEST_QUBIT_IDS


@patch("emu_ct.pulser_adapter.pulser.sampler.sampler.sample")
def test_single_channel(mock_pulser_sample):
    mock_pulser_sample.return_value = make_sequence_samples_mock(
        {
            (10, 20, ("test_qubit_0",)): [(1, 2)] * 10,
            (26, 29, ("test_qubit_2",)): [(3, 4), (5, 6), (7, 8)],
        }
    )

    sampling_rate = 0.5
    res = extract_omega_delta(seq, sampling_rate=sampling_rate)

    expected_number_of_samples = 16  # TEST_DURATION // sampling_rate + 1
    expected = torch.zeros(
        2, expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected[0, 5:10, 0] = 1
    expected[1, 5:10, 0] = 2
    expected[0, 13, 2] = 3
    expected[1, 13, 2] = 4
    expected[0, 14, 2] = 7
    expected[1, 14, 2] = 8

    assert torch.allclose(res, expected)


@patch("emu_ct.pulser_adapter.pulser.sampler.sampler.sample")
def test_autograd(mock_pulser_sample):
    mock_pulser_sample.return_value = make_sequence_samples_mock(
        {
            (10, 20, ("test_qubit_0",)): [(1, 2)] * 10,
            (26, 29, ("test_qubit_2",)): [(3, 4), (5, 6), (7, 8)],
        }
    )

    sampling_rate = 0.5

    res = torch.autograd.grad(
        extract_omega_delta(seq, sampling_rate=sampling_rate)[0, 13, 2].real,
        mock_pulser_sample.return_value.samples_list[0].amp,
    )
    expected = torch.zeros(30)
    expected[26] = 1
    assert torch.allclose(res[0], expected)


@patch("emu_ct.pulser_adapter.pulser.sampler.sampler.sample")
def test_multiple_channels(mock_pulser_sample):
    mock_pulser_sample.return_value = make_sequence_samples_mock(
        {
            (10, 20, ("test_qubit_0",)): [(1, 2)] * 10,
        },
        {
            (
                26,
                29,
                (
                    "test_qubit_0",
                    "test_qubit_2",
                ),
            ): [(3, 4), (5, 6), (7, 8)],
        },
    )

    sampling_rate = 0.5
    res = extract_omega_delta(seq, sampling_rate=sampling_rate)

    expected_number_of_samples = 16  # TEST_DURATION // sampling_rate + 1
    expected = torch.zeros(
        2, expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected[0, 5:10, 0] = 1
    expected[1, 5:10, 0] = 2
    expected[0, 13, (0, 2)] = 3
    expected[1, 13, (0, 2)] = 4
    expected[0, 14, (0, 2)] = 7
    expected[1, 14, (0, 2)] = 8

    assert torch.allclose(res, expected)


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

    sampling_rate = 0.2
    res = extract_omega_delta(seq, sampling_rate=sampling_rate)

    expected_number_of_samples = 7
    expected = torch.zeros(
        2, expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected[0, 2:4, 0] = 1
    expected[1, 2:4, 0] = 2
    expected[0, 2:4, 1:3] = 3
    expected[1, 2:4, 1:3] = 4

    assert torch.allclose(res, expected)


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
        sampling_rate = 0.5
        extract_omega_delta(seq, sampling_rate=sampling_rate)

    assert "multiple pulses acting on same qubit" in str(exception_info.value)
