import torch
import math
import pytest
from unittest.mock import patch, MagicMock
import re

from pulser.backend import EmulationConfig
from pulser.noise_model import NoiseModel
from pulser import Register
from pulser.math import AbstractArray

from emu_base.pulser_adapter import (
    _extract_omega_delta_phi,
    _get_all_lindblad_noise_operators,
    _rydberg_interaction,
    _xy_interaction,
    _get_qubit_positions,
    _get_amp_factors,
    _get_delta_offset,
    PulserData,
    HamiltonianType,
)

TEST_QUBIT_IDS = ["test_qubit_0", "test_qubit_1", "test_qubit_2"]
TEST_C6 = 5420158.53
TEST_C3 = 3700.0

sequence = MagicMock()
sequence.register.qubit_ids = TEST_QUBIT_IDS
sequence.register.qubits = {
    key: AbstractArray(torch.tensor([i, 0.0, 0.0], dtype=torch.float64))
    for i, key in enumerate(TEST_QUBIT_IDS)
}

expected_amp_factors = {
    0: torch.tensor([1.5, 1.0, 0.4]),
    1: torch.tensor([1.5, 1.0, 0.4]),
    2: torch.tensor([1.5, 1.0, 0.4]),
    3: torch.tensor([1.5, 1.0, 0.4]),
    4: torch.tensor([1.5, 1.0, 0.4]),
    5: torch.tensor([1.75, 1.75, 1.75]),
    6: torch.tensor([1.75, 1.75, 1.75]),
    7: torch.tensor([1.75, 1.75, 1.75]),
    8: torch.tensor([1.75, 1.75, 1.75]),
    9: torch.tensor([1.75, 1.75, 1.75]),
    10: torch.tensor([1.75, 1.75, 1.75]),
    11: torch.tensor([1.75, 1.75, 1.75]),
    12: torch.tensor([1.5, 1.0, 0.4]),
}


def mock_sample(hamiltonian_type):
    mock_pulser_dict = {
        hamiltonian_type: {
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
    local_slot1 = MagicMock()
    local_slot1.targets = [TEST_QUBIT_IDS[0]]
    local_slot1.ti = 0
    local_slot1.tf = 5
    local_slot2 = MagicMock()
    local_slot2.targets = [TEST_QUBIT_IDS[0]]
    local_slot2.ti = 12
    local_slot2.tf = 13
    local_slot3 = MagicMock()
    local_slot3.targets = [TEST_QUBIT_IDS[2]]
    local_slot3.ti = 0
    local_slot3.tf = 5
    local_slot4 = MagicMock()
    local_slot4.targets = [TEST_QUBIT_IDS[2]]
    local_slot4.ti = 12
    local_slot4.tf = 13
    global_slot = MagicMock()
    global_slot.targets = TEST_QUBIT_IDS
    global_slot.ti = 5
    global_slot.tf = 12
    local_samples1 = MagicMock()
    local_samples1.slots = [local_slot1, local_slot2]
    local_samples2 = MagicMock()
    local_samples2.slots = [local_slot3, local_slot4]
    global_samples = MagicMock()
    global_samples.slots = [global_slot]
    local_ch_obj1 = MagicMock()
    local_ch_obj1.addressing = "Local"
    local_ch_obj2 = MagicMock()
    local_ch_obj2.addressing = "Local"
    global_ch_obj = MagicMock()
    global_ch_obj.addressing = "Global"
    global_ch_obj.propagation_dir = (0.0, 0.0, 1.0)
    sample_mock = MagicMock()
    sample_mock.to_nested_dict.return_value = {"Local": mock_pulser_dict}
    sample_mock._ch_objs = {
        "local channel 1": local_ch_obj1,
        "local channel 2": local_ch_obj2,
        "global channel": global_ch_obj,
    }
    sample_mock.channel_samples = {
        "local channel 1": local_samples1,
        "local channel 2": local_samples2,
        "global channel": global_samples,
    }
    return sample_mock


@patch("emu_base.pulser_adapter.pulser.sequence.Sequence")
@pytest.mark.parametrize(
    "hamiltonian_type",
    [
        HamiltonianType.Rydberg,
        HamiltonianType.XY,
    ],
)
def test_interaction_coefficient(mock_sequence, hamiltonian_type):
    coords = [
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0],
    ]
    register = Register.from_coordinates(coords, prefix="q")
    qubit_positions = _get_qubit_positions(register)
    assert all(len(q) == 3 for q in qubit_positions)
    assert all(q.dtype == torch.float64 for q in qubit_positions)

    mock_device = MagicMock(interaction_coeff=TEST_C6, interaction_coeff_xy=TEST_C3)
    mock_sequence.register = register
    mock_sequence.device = mock_device
    mock_sequence.magnetic_field = [0.0, 0.0, 30.0]

    if hamiltonian_type == HamiltonianType.Rydberg:
        interaction_matrix = _rydberg_interaction(mock_sequence)
    else:
        interaction_matrix = _xy_interaction(mock_sequence)

    dev = interaction_matrix.device
    dtype = interaction_matrix.dtype
    assert dtype == torch.float64

    if hamiltonian_type == HamiltonianType.Rydberg:
        expected_interaction_matrix = torch.tensor(
            [
                [0.0000, 5.4202, 5.4202 / 64],
                [5.4202, 0.0000, 5.4202],
                [5.4202 / 64, 5.4202, 0.0000],
            ],
            dtype=dtype,
            device=dev,
        )
    else:
        expected_interaction_matrix = torch.tensor(
            [[0.0, 3.7, 3.7 / 8], [3.7, 0.0, 3.7], [3.7 / 8, 3.7, 0.0]],
            dtype=dtype,
            device=dev,
        )

    assert torch.allclose(
        interaction_matrix,
        expected_interaction_matrix,
    )


def test_XY_interaction_with_mag_field():
    coords = [[-8.0, 0.0], [0.0, 0.0], [8.0 * math.sqrt(2 / 3), 8.0 * math.sqrt(1 / 3)]]
    register = Register.from_coordinates(coords, prefix="q")
    mock_device = MagicMock(interaction_coeff_xy=TEST_C3)
    mock_sequence = MagicMock(
        register=register, device=mock_device, magnetic_field=[0.0, 1.0, 0.0]
    )

    interaction_matrix = _xy_interaction(mock_sequence)

    expected_01 = TEST_C3 / 8**3
    r_02 = math.sqrt(2 + 2 * math.sqrt(2 / 3))
    expected_02 = TEST_C3 * (1 - 1 / r_02**2) / (8 * r_02) ** 3
    # element 1,2 is expected to be 0 by the choice of the magnetic field
    expected_interaction_matrix = torch.tensor(
        [
            [0.0, expected_01, expected_02],
            [expected_01, 0.0, 0.0],
            [expected_02, 0.0, 0.0],
        ],
        dtype=interaction_matrix.dtype,
        device=interaction_matrix.device,
    )

    assert torch.allclose(
        interaction_matrix,
        expected_interaction_matrix,
    )


def test_interaction_matrix_differentiability():
    coords = [
        torch.tensor([1.0, 0.1], requires_grad=True),
        torch.tensor([2.0, 0.1], requires_grad=True),
    ]
    register = Register.from_coordinates(coords, prefix="q")
    mock_device = MagicMock(interaction_coeff=TEST_C6, interaction_coeff_xy=TEST_C3)
    mock_sequence = MagicMock(register=register, device=mock_device)

    interaction_matrix = _rydberg_interaction(mock_sequence)
    assert interaction_matrix.requires_grad
    assert not interaction_matrix.is_leaf

    try:
        loss = torch.sum(interaction_matrix)
        torch.autograd.grad(loss, coords)
    except Exception as err:
        raise err


@pytest.mark.parametrize(
    ("hamiltonian_type", "laser_waist"),
    [
        ("ground-rydberg", None),
        ("XY", None),
        ("ground-rydberg", 10.0),
        ("XY", 10.0),
    ],
)
@patch("emu_base.pulser_adapter.torch.normal")
def test_get_amp_factors(mock_randn, hamiltonian_type, laser_waist):
    global_times = list(range(5, 12))
    mock_randn.side_effect = [torch.tensor(1.5), torch.tensor(0.4), torch.tensor(1.75)]
    samples = mock_sample(hamiltonian_type)
    actual = _get_amp_factors(
        samples,
        1.0,
        laser_waist,
        [x.as_tensor() for x in sequence.register.qubits.values()],
        TEST_QUBIT_IDS,
    )
    assert set(actual.keys()) == set(expected_amp_factors.keys())
    for key, val in actual.items():
        if laser_waist is not None and key in global_times:
            waist_amplitudes = torch.tensor(
                [math.exp(-((i / laser_waist) ** 2)) for i in range(3)],
                dtype=torch.float64,
            )
        else:
            waist_amplitudes = torch.ones(3, dtype=torch.float64)
        assert torch.allclose(val, expected_amp_factors[key] * waist_amplitudes)


@patch("emu_base.pulser_adapter.torch.normal")
def test_get_delta_offset(mock_random):
    mock_random.return_value = torch.tensor([1.0, 2.0, 3.0])
    actual = _get_delta_offset(3, 5.0)
    sigma = 0.18978408784721654
    mock_random.assert_called_once_with(0.0, sigma, (3,))
    assert torch.equal(actual, mock_random.return_value)


@pytest.mark.parametrize(
    ("hamiltonian_type", "laser_waist"),
    [
        ("ground-rydberg", None),
        ("XY", None),
        ("ground-rydberg", 10.0),
        ("XY", 10.0),
    ],
)
@patch("emu_base.pulser_adapter._get_qubit_positions")
@patch("emu_base.pulser_adapter.pulser.sampler.sample")
@patch("emu_base.pulser_adapter._get_amp_factors")
@patch("emu_base.pulser_adapter._get_delta_offset")
def test_extract_omega_delta_phi_dt_2(
    mock_offset,
    mock_factors,
    mock_pulser_sample,
    mock_qubit_positions,
    hamiltonian_type,
    laser_waist,
):
    """Local pulse - targe qubit 1:
    pulser.Pulse(RampWaveform(5,3,10),RampWaveform(5,1.5,-10),0.1) and
    Global pulse: Pulse(RampWaveform(8,10.0,0.0),RampWaveform(8,-10,10),0.2)"""
    TEST_DURATION = 13
    dt = 2
    target_times = torch.arange(0, TEST_DURATION + 1, dt).tolist()
    sequence.get_duration.return_value = TEST_DURATION

    mock_qubit_positions.return_value = [
        torch.tensor([-1, 0, 1], dtype=torch.float64),
        torch.tensor([1, 0, 1], dtype=torch.float64),
        torch.tensor([2, 0, 1], dtype=torch.float64),
    ]
    mock_offset.return_value = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    mock_pulser_sample.return_value = mock_sample(hamiltonian_type)
    mock_factors.return_value = expected_amp_factors

    actual_omega, actual_delta, actual_phi = _extract_omega_delta_phi(
        sequence=sequence,
        target_times=target_times,
        with_modulation=False,
        laser_waist=laser_waist,
        amp_sigma=1.0,
        temperature=5.0,
    )

    expected_number_of_samples = math.ceil(TEST_DURATION / dt - 0.5)
    assert len(actual_omega) == expected_number_of_samples

    expected_omega = torch.tensor(
        [
            [4.75, 0.0, 4.75],
            [8.25, 0.0, 8.25],
            [10.0, 10.0, 10.0],
            [7.1429, 7.1429, 7.1429],
            [4.2857, 4.2857, 4.2857],
            [1.4286, 1.4286, 1.4286],
        ],
        dtype=torch.complex128,
    )
    for i in range(6):
        to_modify = expected_omega[i]
        to_modify *= expected_amp_factors[2 * i + 1]
    expected_delta = torch.tensor(
        [
            [-1.3750, 0.0000, 0.0000],
            [-7.1250, 0.0000, 0.0000],
            [-10.0000, -10.0000, -10.0000],
            [-4.2857, -4.2857, -4.2857],
            [1.4286, 1.4286, 1.4286],
            [7.1429, 7.1429, 7.1429],
        ],
        dtype=torch.complex128,
    )
    expected_phi = torch.tensor(
        [
            [0.1000, 0.0000, 0.0000],
            [0.1000, 0.0000, 0.0000],
            [0.2000, 0.2000, 0.2000],
            [0.2000, 0.2000, 0.2000],
            [0.2000, 0.2000, 0.2000],
            [0.2000, 0.2000, 0.2000],
        ],
        dtype=torch.complex128,
    )
    mock_factors.assert_called_once_with(
        mock_pulser_sample.return_value,
        1.0,
        laser_waist,
        mock_qubit_positions.return_value,
        TEST_QUBIT_IDS,
    )
    mock_offset.assert_called_once_with(3, 5.0)
    assert torch.allclose(actual_omega, expected_omega, rtol=0, atol=1e-4)
    assert torch.allclose(
        actual_delta, expected_delta + mock_offset.return_value, rtol=0, atol=1e-4
    )
    assert torch.allclose(actual_phi, expected_phi, rtol=0, atol=1e-4)


@pytest.mark.parametrize(
    ("hamiltonian_type", "laser_waist"),
    [
        ("ground-rydberg", None),
        ("XY", None),
        ("ground-rydberg", 10.0),
        ("XY", 10.0),
    ],
)
@patch("emu_base.pulser_adapter._get_qubit_positions")
@patch("emu_base.pulser_adapter.pulser.sampler.sample")
@patch("emu_base.pulser_adapter._get_amp_factors")
@patch("emu_base.pulser_adapter._get_delta_offset")
def test_extract_omega_delta_phi_dt_1(
    mock_offset,
    mock_factors,
    mock_pulser_sample,
    mock_qubit_positions,
    hamiltonian_type,
    laser_waist,
):
    """Local pulse - targe qubit 1:
    pulser.Pulse(RampWaveform(5,3,10),RampWaveform(5,1.5,-10),0.1) and
    Global pulse: Pulse(RampWaveform(8,10.0,0.0),RampWaveform(8,-10,10),0.2)"""
    TEST_DURATION = 13
    dt = 1
    target_times = torch.arange(0, TEST_DURATION + 1, dt).tolist()
    sequence.get_duration.return_value = TEST_DURATION
    mock_qubit_positions.return_value = [
        torch.tensor([-1, 0, 1], dtype=torch.float64),
        torch.tensor([1, 0, 1], dtype=torch.float64),
        torch.tensor([2, 0, 1], dtype=torch.float64),
    ]

    mock_offset.return_value = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    mock_pulser_sample.return_value = mock_sample(hamiltonian_type)
    mock_factors.return_value = expected_amp_factors

    actual_omega, actual_delta, actual_phi = _extract_omega_delta_phi(
        sequence=sequence,
        target_times=target_times,
        with_modulation=False,
        laser_waist=laser_waist,
        amp_sigma=1.0,
        temperature=5.0,
    )

    expected_number_of_samples = math.ceil(TEST_DURATION / dt - 0.5)
    assert len(actual_omega) == expected_number_of_samples

    expected_omega = torch.tensor(
        [
            [
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
            [
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
            [
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
        ],
        dtype=torch.complex128,
    ).T
    for i in range(12):
        expected_omega[i] = 0.5 * (
            expected_omega[i] * expected_amp_factors[i]
            + expected_omega[i + 1] * expected_amp_factors[i + 1]
        )
    # the element omega[4,0] should not simply be multiplied by wais_amplitudes
    # it is the average of two samples, one of which should be multiplied, and the other not
    # this test has different qubit positions than the dt=2 one to test precisely this.
    expected_delta = torch.tensor(
        [
            [
                0.0625,
                -2.8125,
                -5.6875,
                -8.5625,
                -10.0,
                -8.57142857,
                -5.714285715,
                -2.8571428599999997,
                0.0,
                2.8571428599999997,
                5.714285715,
                8.57142857,
                11.42857143,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                -5.0,
                -8.57142857,
                -5.714285715,
                -2.8571428599999997,
                0.0,
                2.8571428599999997,
                5.714285715,
                8.57142857,
                11.42857143,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                -5.0,
                -8.57142857,
                -5.714285715,
                -2.8571428599999997,
                0.0,
                2.8571428599999997,
                5.714285715,
                8.57142857,
                11.42857143,
            ],
        ],
        dtype=torch.complex128,
    ).T
    expected_phi = torch.tensor(
        [
            [0.1000 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [0.1000 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [0.1000 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [0.1000 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [0.1500 + 0.0j, 0.1000 + 0.0j, 0.1000 + 0.0j],
            [0.2000 + 0.0j, 0.2000 + 0.0j, 0.2000 + 0.0j],
            [0.2000 + 0.0j, 0.2000 + 0.0j, 0.2000 + 0.0j],
            [0.2000 + 0.0j, 0.2000 + 0.0j, 0.2000 + 0.0j],
            [0.2000 + 0.0j, 0.2000 + 0.0j, 0.2000 + 0.0j],
            [0.2000 + 0.0j, 0.2000 + 0.0j, 0.2000 + 0.0j],
            [0.2000 + 0.0j, 0.2000 + 0.0j, 0.2000 + 0.0j],
            [0.2000 + 0.0j, 0.2000 + 0.0j, 0.2000 + 0.0j],
            [0.2000 + 0.0j, 0.2000 + 0.0j, 0.2000 + 0.0j],
        ],
        dtype=torch.complex128,
    )

    mock_factors.assert_called_once_with(
        mock_pulser_sample.return_value,
        1.0,
        laser_waist,
        mock_qubit_positions.return_value,
        TEST_QUBIT_IDS,
    )
    mock_offset.assert_called_once_with(3, 5.0)
    assert torch.allclose(actual_omega, expected_omega, rtol=0, atol=1e-4)
    assert torch.allclose(
        actual_delta, expected_delta + mock_offset.return_value, rtol=0, atol=1e-4
    )
    assert torch.allclose(actual_phi, expected_phi, rtol=0, atol=1e-4)


@patch("emu_base.pulser_adapter.pulser.sampler.sample")
def test_autograd(mock_pulser_sample):
    TEST_DURATION = 10
    dt = 2
    target_times = torch.arange(0, TEST_DURATION + 1, dt).tolist()
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
    omega_value = _extract_omega_delta_phi(
        sequence=sequence,
        target_times=target_times,
        with_modulation=False,
        laser_waist=None,
        amp_sigma=0.0,
        temperature=0.0,
    )[0][2, 2].real

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

    assert _get_all_lindblad_noise_operators(noise_model) == []


def test_get_all_lindblad_operators():
    random_collapse = torch.rand(2, 2, dtype=torch.complex128)

    noise_model = NoiseModel(
        depolarizing_rate=0.16,
        dephasing_rate=0.005,
        eff_noise_rates=(0.0036,),
        eff_noise_opers=(random_collapse,),
    )

    ops = _get_all_lindblad_noise_operators(noise_model)

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


@patch("emu_base.pulser_adapter.pulser.sampler.sample")
def test_parsed_sequence(mock_pulser_sample):
    TEST_DURATION = 10
    dt = 2
    target_times = torch.arange(0, TEST_DURATION + 1, dt).tolist()
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

    adressed_basis = "XY"
    sequence.get_addressed_bases.return_value = [adressed_basis]

    mock_pulser_dict = {
        adressed_basis: {
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

    interaction_matrix = [
        [0.0, 0.0929, -0.4],
        [0.0929, 0.0, 0.1067],
        [-0.4, 0.1067, 0.0],
    ]
    sequence._slm_mask_time = []

    random_collapse = torch.rand(2, 2, dtype=torch.complex128)
    noise_model = NoiseModel(
        depolarizing_rate=0.16,
        dephasing_rate=0.005,
        eff_noise_rates=(0.0036,),
        eff_noise_opers=(random_collapse,),
    )

    ops = _get_all_lindblad_noise_operators(noise_model)

    config = EmulationConfig(
        noise_model=noise_model,
        interaction_matrix=interaction_matrix,
        interaction_cutoff=0.15,
    )

    parsed_sequence = PulserData(sequence=sequence, config=config, dt=dt)
    omega, delta, phi = _extract_omega_delta_phi(
        sequence=sequence,
        target_times=target_times,
        with_modulation=False,
        laser_waist=None,
        amp_sigma=0.0,
        temperature=0.0,
    )

    cutoff_interaction_matrix = torch.tensor(
        [[0, 0, -0.4], [0, 0, 0], [-0.4, 0, 0]],
        dtype=torch.float64,
    )

    assert torch.allclose(parsed_sequence.omega, omega)
    assert torch.allclose(parsed_sequence.delta, delta)
    assert torch.allclose(parsed_sequence.phi, phi)
    assert torch.allclose(
        parsed_sequence.full_interaction_matrix, cutoff_interaction_matrix
    )
    assert torch.allclose(
        parsed_sequence.masked_interaction_matrix, cutoff_interaction_matrix
    )
    assert parsed_sequence.slm_end_time == 0.0
    assert parsed_sequence.hamiltonian_type == HamiltonianType.XY
    assert len(parsed_sequence.lindblad_ops) == len(ops)
    for i in range(len(ops)):
        assert torch.allclose(ops[i], parsed_sequence.lindblad_ops[i])

    sequence._slm_mask_time = [1.0, 10.0]
    sequence._slm_mask_targets = [1]
    masked_interaction_matrix = cutoff_interaction_matrix.clone().detach()

    parsed_sequence = PulserData(sequence=sequence, config=config, dt=dt)
    omega, delta, phi = _extract_omega_delta_phi(
        sequence=sequence,
        target_times=target_times,
        with_modulation=False,
        laser_waist=None,
        amp_sigma=0.0,
        temperature=0.0,
    )

    assert torch.allclose(parsed_sequence.omega, omega)
    assert torch.allclose(parsed_sequence.delta, delta)
    assert torch.allclose(parsed_sequence.phi, phi)
    assert torch.allclose(
        parsed_sequence.full_interaction_matrix, cutoff_interaction_matrix
    )
    assert torch.allclose(
        parsed_sequence.masked_interaction_matrix, masked_interaction_matrix
    )
    assert parsed_sequence.slm_end_time == 10.0
    assert parsed_sequence.hamiltonian_type == HamiltonianType.XY
    assert len(parsed_sequence.lindblad_ops) == len(ops)
    for i in range(len(ops)):
        assert torch.allclose(ops[i], parsed_sequence.lindblad_ops[i])


@patch("emu_base.pulser_adapter._get_qubit_positions")
@patch("emu_base.pulser_adapter.pulser.sampler.sample")
def test_laser_waist(mock_pulser_sample, mock_qubit_positions):
    mock_qubit_positions.return_value = [
        torch.tensor([i, 0, 1], dtype=torch.float64) for i in range(3)
    ]
    TEST_DURATION = 10
    dt = 2
    target_times = torch.arange(0, TEST_DURATION + 1, dt).tolist()
    sequence.get_duration.return_value = TEST_DURATION
    adressed_basis = "ground-rydberg"
    sequence.get_addressed_bases.return_value = [adressed_basis]
    mock_pulser_sample.return_value = mock_sample(adressed_basis)

    mat = torch.randn(3, 3, dtype=float)
    interaction_matrix = (mat + mat.T).fill_diagonal_(0).tolist()

    sequence._slm_mask_time = []

    noise_model = NoiseModel(
        laser_waist=0.1, amp_sigma=1.0, runs=1, samples_per_run=1, temperature=5.0
    )

    config = EmulationConfig(
        noise_model=noise_model,
        interaction_matrix=interaction_matrix,
        interaction_cutoff=0.15,
    )
    torch.manual_seed(1337)
    parsed_sequence = PulserData(sequence=sequence, config=config, dt=dt)
    torch.manual_seed(1337)
    omega, delta, phi = _extract_omega_delta_phi(
        sequence=sequence,
        target_times=target_times,
        with_modulation=False,
        laser_waist=0.1,
        amp_sigma=1.0,
        temperature=5.0,
    )
    assert torch.allclose(omega, parsed_sequence.omega)
    assert torch.allclose(delta, parsed_sequence.delta)
    assert torch.allclose(phi, parsed_sequence.phi)


@pytest.mark.parametrize(
    "hamiltonian_type",
    ["ground-rydberg", "XY"],
)
@patch("emu_base.pulser_adapter._get_qubit_positions")
@patch("emu_base.pulser_adapter.pulser.sampler.sample")
def test_supported_noise_types(
    mock_pulser_sample, mock_qubit_positions, hamiltonian_type
):
    mock_qubit_positions.return_value = [
        torch.tensor([i, 0, 1], dtype=torch.float64) for i in range(3)
    ]
    TEST_DURATION = 10
    dt = 2
    sequence.get_duration.return_value = TEST_DURATION
    sequence.get_addressed_bases.return_value = [hamiltonian_type]
    mock_pulser_sample.return_value = mock_sample(hamiltonian_type)

    mat = torch.randn(3, 3, dtype=float)
    interaction_matrix = (mat + mat.T).fill_diagonal_(0).tolist()

    sequence._slm_mask_time = []

    noise_model = NoiseModel(
        laser_waist=0.1,
        amp_sigma=1.0,
        runs=1,
        samples_per_run=1,
        temperature=5.0,
        state_prep_error=0.5,
        p_false_neg=0.1,
        p_false_pos=0.2,
        relaxation_rate=5.0,
        dephasing_rate=11.0,
        # hyperfine_dephasing_rate= 12., not supported yet
        depolarizing_rate=13.0,
        with_leakage=True,
        eff_noise_rates=(0.1,),
        eff_noise_opers=(torch.randn(3, 3, dtype=torch.float64),),
    )

    config = EmulationConfig(
        noise_model=noise_model,
        interaction_matrix=interaction_matrix,
        interaction_cutoff=0.15,
    )
    torch.manual_seed(1337)
    if hamiltonian_type == "XY":
        unsupported_noises = {"doppler", "relaxation", "amplitude", "leakage"}
        regex_string = (
            "Interaction mode 'HamiltonianType.XY'"
            " does not support simulation of noise types:"
            r" (\w+), (\w+), (\w+), (\w+)."
        )
    elif hamiltonian_type == "ground-rydberg":
        unsupported_noises = {"leakage"}
        regex_string = (
            "Interaction mode 'HamiltonianType.Rydberg'"
            " does not support simulation of noise types:"
            r" (\w+)."
        )
    else:
        assert False, "bad hamiltonian type"
    with pytest.raises(NotImplementedError) as exc:
        PulserData(sequence=sequence, config=config, dt=dt)
    # the noise types do not show in the exception in a deterministic way
    # regex to the rescue
    match = re.match(regex_string, str(exc.value))
    assert set(match.groups()) == unsupported_noises
