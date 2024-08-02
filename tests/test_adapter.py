from emu_mps.pulser_adapter import (
    extract_omega_delta_phi,
    get_all_lindblad_noise_operators,
)
from unittest.mock import patch, MagicMock
import pytest
import torch
from pulser.noise_model import NoiseModel

TEST_DURATION = 31
TEST_QUBIT_IDS = ["test_qubit_0", "test_qubit_1", "test_qubit_2"]


def make_channel_samples_mock(channels_data):
    channel_samples = MagicMock()
    channel_samples.amp = torch.zeros(TEST_DURATION)
    channel_samples.det = torch.zeros(TEST_DURATION)
    channel_samples.phase = torch.zeros(TEST_DURATION)
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
        channel_samples.phase[slot.ti : slot.tf] = torch.Tensor(
            [x[2] for x in slot_samples]
        )
        slot.targets = slot_data[2]
        channel_samples.slots.append(slot)

    channel_samples.amp.requires_grad = True
    channel_samples.det.requires_grad = True
    channel_samples.phase.requires_grad = True

    return channel_samples


def make_sequence_samples_mock(*channels):
    """
    Creates a mock samples sequence, specified by channels containing slots and samples.
    Each channel should be a dictionary
        { (ti, tf, (qubits, ...)): [(amp0, det0,phase0), (amp1, det1,phase1), ...], ...}
    where
        ti: time at beginning of slot
        tf: time at end of slot (tf is not included in the slot)
        qubits: qubit id strings
        amp0, amp1, ...: amplitude samples for that slot
        det0, det1, ...: detuning samples for that slot
        phase0, phase1, ...: phase camples for that slot


    Slots don't have to be consecutive.
    """
    sequence_samples = MagicMock()
    sequence_samples.samples_list = [make_channel_samples_mock(c) for c in channels]
    sequence_samples.max_duration = TEST_DURATION

    return sequence_samples


sequence = MagicMock()
sequence.register.qubit_ids = TEST_QUBIT_IDS


@patch("emu_mps.pulser_adapter.pulser.sampler.sampler.sample")
def test_single_channel(mock_pulser_sample):
    mock_pulser_sample.return_value = make_sequence_samples_mock(
        {
            (1, 2, ("test_qubit_1",)): [(2, -1, 0.1)],
            (10, 20, ("test_qubit_0",)): [(1, -2, 0.2)] * 10,
            (26, 30, ("test_qubit_2",)): [
                (3, 4, 1.1),
                (5, 6, 1.2),
                (7, 8, 1.3),
                (9, 10, 1.4),
            ],
        }
    )

    dt = 2
    actual_omega, actual_delta, actual_phi = extract_omega_delta_phi(sequence, dt, False)

    expected_number_of_samples = 15  # ceil(TEST_DURATION / sampling_rate -0.5)
    expected_omega = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_delta = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_phi = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_omega[0, 1] = 2
    expected_delta[0, 1] = -1
    expected_phi[0, 1] = 0.1

    expected_omega[5:10, 0] = 1
    expected_delta[5:10, 0] = -2
    expected_phi[5:10, 0] = 0.2

    expected_omega[13, 2] = 5
    expected_delta[13, 2] = 6
    expected_phi[13, 2] = 1.2

    expected_omega[14, 2] = 9
    expected_delta[14, 2] = 10
    expected_phi[14, 2] = 1.4

    assert torch.allclose(actual_omega, expected_omega)
    assert torch.allclose(actual_delta, expected_delta)
    assert torch.allclose(actual_phi, expected_phi)


@patch("emu_mps.pulser_adapter.pulser.sampler.sampler.sample")
def test_autograd(mock_pulser_sample):
    mock_pulser_sample.return_value = make_sequence_samples_mock(
        {
            (10, 20, ("test_qubit_0",)): [(1, 2, 0.0)] * 10,
            (26, 29, ("test_qubit_2",)): [(3, 4, 0.5), (5, 6, 0.6), (7, 8, 0.8)],
        }
    )
    dt = 2
    res = torch.autograd.grad(
        extract_omega_delta_phi(sequence, dt, False)[0][13, 2].real,
        mock_pulser_sample.return_value.samples_list[0].amp,
    )
    expected = torch.zeros(31)
    expected[27] = 1
    assert torch.allclose(res[0], expected)


@patch("emu_mps.pulser_adapter.pulser.sampler.sampler.sample")
def test_multiple_channels(mock_pulser_sample):
    mock_pulser_sample.return_value = make_sequence_samples_mock(
        {
            (10, 20, ("test_qubit_0",)): [(1, -2, 0)] * 10,
        },
        {
            (
                26,
                30,
                (
                    "test_qubit_0",
                    "test_qubit_2",
                ),
            ): [(3, 4, 0.5), (5, 6, 0.6), (7, 8, 0.7), (9, 10, 0.8)],
        },
    )
    dt = 2
    actual_omega, actual_delta, actual_phi = extract_omega_delta_phi(sequence, dt, False)
    expected_number_of_samples = 15  # TEST_DURATION // sampling_rate + 1
    expected_omega = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_delta = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_phi = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_omega[5:10, 0] = 1
    expected_delta[5:10, 0] = -2
    expected_phi[5:10, 0] = 0

    expected_omega[13, (0, 2)] = 5
    expected_delta[13, (0, 2)] = 6
    expected_phi[13, (0, 2)] = 0.6

    expected_omega[14, (0, 2)] = 9
    expected_delta[14, (0, 2)] = 10
    expected_phi[14, (0, 2)] = 0.8

    assert torch.allclose(actual_omega, expected_omega)
    assert torch.allclose(actual_delta, expected_delta)
    assert torch.allclose(actual_phi, expected_phi)


@patch("emu_mps.pulser_adapter.pulser.sampler.sampler.sample")
def test_multiple_channels_together(mock_pulser_sample):
    mock_pulser_sample.return_value = make_sequence_samples_mock(
        {
            (10, 20, ("test_qubit_0",)): [(1, 2, 0.0)] * 10,
        },
        {
            (10, 20, ("test_qubit_1", "test_qubit_2")): [(3, 4, 0.0)] * 10,
        },
    )
    dt = 5

    actual_omega, actual_delta, actual_phi = extract_omega_delta_phi(sequence, dt, False)

    expected_number_of_samples = 6
    expected_omega = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )
    expected_delta = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )

    expected_phi = torch.zeros(
        expected_number_of_samples, len(TEST_QUBIT_IDS), dtype=torch.complex128
    )

    expected_omega[2:4, 0] = 1
    expected_delta[2:4, 0] = 2
    expected_phi[2:4, 0] = 0

    expected_omega[2:4, 1:3] = 3
    expected_delta[2:4, 1:3] = 4
    expected_phi[2:4, 1:3] = 0

    assert torch.allclose(actual_omega, expected_omega)
    assert torch.allclose(actual_delta, expected_delta)
    assert torch.allclose(actual_phi, expected_phi)


@patch("emu_mps.pulser_adapter.pulser.sampler.sampler.sample")
def test_multiple_channels_together_on_same_qubit(mock_pulser_sample):
    mock_pulser_sample.return_value = make_sequence_samples_mock(
        {
            (10, 20, ("test_qubit_0",)): [(1, 2, 0)] * 10,
        },
        {
            (10, 20, ("test_qubit_0", "test_qubit_1")): [(1, 2, 0)] * 10,
        },
    )
    with pytest.raises(NotImplementedError) as exception_info:
        dt = 2
        extract_omega_delta_phi(sequence, dt, False)
    assert "multiple pulses acting on same qubit" in str(exception_info.value)


def test_get_all_lindblad_operators_no_noise():
    noise_model = NoiseModel()

    assert get_all_lindblad_noise_operators(noise_model) == []


def test_get_all_lindblad_operators():
    random_collapse = torch.rand(2, 2, dtype=torch.complex128)

    noise_model = NoiseModel(
        # FIXME: remove below line when Pulser is upgraded
        # past 1b3735df935f8ee37fcaee5055b40e801d794466
        noise_types=("depolarizing", "dephasing", "eff_noise"),
        depolarizing_rate=0.16,
        dephasing_rate=0.005,
        # FIXME: remove below line when Pulser is upgraded
        # past 1b3735df935f8ee37fcaee5055b40e801d794466
        hyperfine_dephasing_rate=0.0,
        eff_noise_rates=(0.0036,),
        eff_noise_opers=(random_collapse,),
    )

    ops = get_all_lindblad_noise_operators(noise_model)

    assert len(ops) == 5

    # Depolarizing
    assert torch.allclose(
        ops[0],
        torch.tensor(
            [
                [0, 0.2],
                [0.2, 0],
            ],
            dtype=torch.complex128,
        ),
    )

    assert torch.allclose(
        ops[1],
        torch.tensor(
            [
                [0, 0.2j],
                [-0.2j, 0],
            ],
            dtype=torch.complex128,
        ),
    )

    assert torch.allclose(
        ops[2],
        torch.tensor(
            [
                [-0.2, 0],
                [0, 0.2],
            ],
            dtype=torch.complex128,
        ),
    )

    # Dephasing
    assert torch.allclose(
        ops[3],
        torch.tensor(
            [
                [-0.05, 0],
                [0, 0.05],
            ],
            dtype=torch.complex128,
        ),
    )

    # Effective noise
    assert torch.allclose(
        torch.tensor(
            [
                [ops[4][1, 1], ops[4][1, 0]],
                [ops[4][0, 1], ops[4][0, 0]],
            ]
        ),
        0.06 * random_collapse,
    )
