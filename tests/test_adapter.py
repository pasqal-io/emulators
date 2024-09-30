from emu_mps.pulser_adapter import (
    extract_omega_delta_phi,
    get_all_lindblad_noise_operators,
)
from unittest.mock import patch, MagicMock

import torch
from pulser.noise_model import NoiseModel
import math


TEST_QUBIT_IDS = ["test_qubit_0", "test_qubit_1", "test_qubit_2"]

sequence = MagicMock()
sequence.register.qubit_ids = TEST_QUBIT_IDS


@patch("emu_mps.pulser_adapter.pulser.sampler.sample")
def test_global_channel(mock_pulser_sample):
    """Global pulse: Pulse(RampWaveform(10,10.0,0.0),RampWaveform(10,-10,10),0.2)"""
    TEST_DURATION = 10
    dt = 2
    sequence.get_duration.return_value = TEST_DURATION
    amp_tensor = torch.tensor(
        [
            10.0,
            8.88888889,
            7.77777778,
            6.66666667,
            5.55555556,
            4.44444444,
            3.33333333,
            2.22222222,
            1.11111111,
            0.0,
        ],
        dtype=torch.complex128,
    )
    det_tensor = torch.tensor(
        [
            -10.0,
            -7.77777778,
            -5.55555556,
            -3.33333333,
            -1.11111111,
            1.11111111,
            3.33333333,
            5.55555556,
            7.77777778,
            10.0,
        ],
        dtype=torch.complex128,
    )
    phase_tensor = torch.tensor([0.2] * 10, dtype=torch.complex128)

    mock_pulser_dict = {
        "ground-rydberg": {
            TEST_QUBIT_IDS[1]: {
                "amp": amp_tensor,
                "det": det_tensor,
                "phase": phase_tensor,
            },
            TEST_QUBIT_IDS[2]: {
                "amp": amp_tensor,
                "det": det_tensor,
                "phase": phase_tensor,
            },
            TEST_QUBIT_IDS[0]: {
                "amp": amp_tensor,
                "det": det_tensor,
                "phase": phase_tensor,
            },
        }
    }

    sample_instance = MagicMock()
    sample_instance.to_nested_dict.return_value = {"Local": mock_pulser_dict}
    mock_pulser_sample.return_value = sample_instance

    actual_omega, actual_delta, actual_phi = extract_omega_delta_phi(sequence, dt, False)

    expected_number_of_samples = math.ceil(TEST_DURATION / dt - 0.5)

    assert len(actual_omega) == expected_number_of_samples

    expected_omega = (amp_tensor.unsqueeze(1).repeat(1, 3))[1::2]
    expected_delta = (det_tensor.unsqueeze(1).repeat(1, 3))[1::2]
    expected_phi = (phase_tensor.unsqueeze(1).repeat(1, 3))[1::2]

    assert torch.allclose(actual_omega, expected_omega, atol=1e-5)
    assert torch.allclose(actual_delta, expected_delta, atol=1e-5)
    assert torch.allclose(actual_phi, expected_phi, atol=1e-5)


@patch("emu_mps.pulser_adapter.pulser.sampler.sample")
def test_local_global_channel(mock_pulser_sample):
    """Local pulse - targe qubit 1:
    pulser.Pulse(RampWaveform(5,3,10),RampWaveform(5,1.5,-10),0.1) and
    Global pulse: Pulse(RampWaveform(8,10.0,0.0),RampWaveform(8,-10,10),0.2)"""
    TEST_DURATION = 13
    dt = 2
    sequence.get_duration.return_value = TEST_DURATION

    mock_pulser_dict = {
        "ground-rydberg": {
            TEST_QUBIT_IDS[1]: {
                "amp": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    10.0,
                    8.57142857,
                    7.14285714,
                    5.71428571,
                    4.28571429,
                    2.85714286,
                    1.42857143,
                    0.0,
                ],
                "det": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -10.0,
                    -7.14285714,
                    -4.28571429,
                    -1.42857143,
                    1.42857143,
                    4.28571429,
                    7.14285714,
                    10.0,
                ],
                "phase": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                ],
            },
            TEST_QUBIT_IDS[2]: {
                "amp": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    10.0,
                    8.57142857,
                    7.14285714,
                    5.71428571,
                    4.28571429,
                    2.85714286,
                    1.42857143,
                    0.0,
                ],
                "det": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -10.0,
                    -7.14285714,
                    -4.28571429,
                    -1.42857143,
                    1.42857143,
                    4.28571429,
                    7.14285714,
                    10.0,
                ],
                "phase": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                ],
            },
            TEST_QUBIT_IDS[0]: {
                "amp": [
                    3.0,
                    4.75,
                    6.5,
                    8.25,
                    10.0,
                    10.0,
                    8.57142857,
                    7.14285714,
                    5.71428571,
                    4.28571429,
                    2.85714286,
                    1.42857143,
                    0.0,
                ],
                "det": [
                    1.5,
                    -1.375,
                    -4.25,
                    -7.125,
                    -10.0,
                    -10.0,
                    -7.14285714,
                    -4.28571429,
                    -1.42857143,
                    1.42857143,
                    4.28571429,
                    7.14285714,
                    10.0,
                ],
                "phase": [
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                ],
            },
        }
    }

    sample_mock = MagicMock()
    sample_mock.to_nested_dict.return_value = {"Local": mock_pulser_dict}
    mock_pulser_sample.return_value = sample_mock

    actual_omega, actual_delta, actual_phi = extract_omega_delta_phi(sequence, dt, False)

    expected_number_of_samples = math.ceil(TEST_DURATION / dt - 0.5)
    assert len(actual_omega) == expected_number_of_samples

    expected_omega = torch.tensor(
        [
            [4.7500 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [8.2500 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [10.0000 + 0.0j, 10.0000 + 0.0j, 10.0000 + 0.0j],
            [7.1429 + 0.0j, 7.1429 + 0.0j, 7.1429 + 0.0j],
            [4.2857 + 0.0j, 4.2857 + 0.0j, 4.2857 + 0.0j],
            [1.4286 + 0.0j, 1.4286 + 0.0j, 1.4286 + 0.0j],
        ],
        dtype=torch.complex128,
    )
    expected_delta = torch.tensor(
        [
            [-1.3750 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [-7.1250 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [-10.0000 + 0.0j, -10.0000 + 0.0j, -10.0000 + 0.0j],
            [-4.2857 + 0.0j, -4.2857 + 0.0j, -4.2857 + 0.0j],
            [1.4286 + 0.0j, 1.4286 + 0.0j, 1.4286 + 0.0j],
            [7.1429 + 0.0j, 7.1429 + 0.0j, 7.1429 + 0.0j],
        ],
        dtype=torch.complex128,
    )
    expected_phi = torch.tensor(
        [
            [0.1000 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [0.1000 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [0.2000 + 0.0j, 0.2000 + 0.0j, 0.2000 + 0.0j],
            [0.2000 + 0.0j, 0.2000 + 0.0j, 0.2000 + 0.0j],
            [0.2000 + 0.0j, 0.2000 + 0.0j, 0.2000 + 0.0j],
            [0.2000 + 0.0j, 0.2000 + 0.0j, 0.2000 + 0.0j],
        ],
        dtype=torch.complex128,
    )

    assert torch.allclose(actual_omega, expected_omega, atol=1e-4)

    assert torch.allclose(actual_delta, expected_delta, atol=1e-4)
    assert torch.allclose(actual_phi, expected_phi, atol=1e-4)


@patch("emu_mps.pulser_adapter.pulser.sampler.sample")
def test_autograd(mock_pulser_sample):
    TEST_DURATION = 10
    dt = 2
    sequence.get_duration.return_value = TEST_DURATION
    amp_tensor = torch.tensor(
        [
            10.0,
            8.88888889,
            7.77777778,
            6.66666667,
            5.55555556,
            4.44444444,
            3.33333333,
            2.22222222,
            1.11111111,
            0.0,
        ],
        dtype=torch.complex128,
        requires_grad=True,
    )
    det_tensor = torch.tensor(
        [
            -10.0,
            -7.77777778,
            -5.55555556,
            -3.33333333,
            -1.11111111,
            1.11111111,
            3.33333333,
            5.55555556,
            7.77777778,
            10.0,
        ],
        dtype=torch.complex128,
        requires_grad=True,
    )
    phase_tensor = torch.tensor(
        [0.2] * 10,
        dtype=torch.complex128,
        requires_grad=True,
    )

    mock_pulser_dict = {
        "ground-rydberg": {
            TEST_QUBIT_IDS[1]: {
                "amp": amp_tensor,
                "det": det_tensor,
                "phase": phase_tensor,
            },
            TEST_QUBIT_IDS[2]: {
                "amp": amp_tensor,
                "det": det_tensor,
                "phase": phase_tensor,
            },
            TEST_QUBIT_IDS[0]: {
                "amp": amp_tensor,
                "det": det_tensor,
                "phase": phase_tensor,
            },
        }
    }

    # the sequence
    sample_mock = MagicMock()
    sample_mock.to_nested_dict.return_value = {"Local": mock_pulser_dict}
    mock_pulser_sample.return_value = sample_mock

    # first data for grad
    omega_value = extract_omega_delta_phi(sequence, dt, False)[0][2, 2].real

    # second data to for grad
    dict_sample = sample_mock.to_nested_dict()
    dict_sample_atom2 = dict_sample["Local"]["ground-rydberg"][TEST_QUBIT_IDS[2]]["amp"]

    res = torch.autograd.grad(
        omega_value,
        dict_sample_atom2,
    )
    expected = torch.zeros(TEST_DURATION, dtype=torch.complex128)
    expected[5] = 1
    assert torch.allclose(res[0], expected)


def test_get_all_lindblad_operators_no_noise():
    noise_model = NoiseModel()

    assert get_all_lindblad_noise_operators(noise_model) == []


def test_get_all_lindblad_operators():
    random_collapse = torch.rand(2, 2, dtype=torch.complex128)

    noise_model = NoiseModel(
        depolarizing_rate=0.16,
        dephasing_rate=0.005,
        eff_noise_rates=(0.0036,),
        eff_noise_opers=(random_collapse,),
    )

    ops = get_all_lindblad_noise_operators(noise_model)

    assert len(ops) == 5

    # Depolarizing
    assert torch.allclose(
        ops[1],
        torch.tensor(
            [
                [0, 0.2],
                [0.2, 0],
            ],
            dtype=torch.complex128,
        ),
    )

    assert torch.allclose(
        ops[2],
        torch.tensor(
            [
                [0, 0.2j],
                [-0.2j, 0],
            ],
            dtype=torch.complex128,
        ),
    )

    assert torch.allclose(
        ops[3],
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
        ops[0],
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
