import torch
import math
import pytest
from unittest.mock import patch, MagicMock

import pulser
from pulser.backend import EmulationConfig, Observable
from pulser.noise_model import NoiseModel
from pulser.math import AbstractArray

from emu_base.pulser_adapter import (
    _extract_omega_delta_phi,
    _get_all_lindblad_noise_operators,
    _get_target_times,
    PulserData,
    HamiltonianType,
)

dtype = torch.complex128

TEST_QUBIT_IDS = ["test_qubit_0", "test_qubit_1", "test_qubit_2"]
TEST_C6 = 5420158.53
TEST_C3 = 3700.0

sequence = MagicMock()
sequence.register.qubit_ids = TEST_QUBIT_IDS
sequence.register.qubits = {
    key: AbstractArray(torch.tensor([i, 0.0, 0.0], dtype=torch.float64))
    for i, key in enumerate(TEST_QUBIT_IDS)
}

mock_observable = MagicMock(spec=Observable, evaluation_times=None)

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


@pytest.mark.parametrize(
    ("hamiltonian_type"),
    [
        "ground-rydberg",
        "XY",
    ],
)
@patch("emu_base.pulser_adapter.HamiltonianData")
def test_extract_omega_delta_phi_dt_2(
    mock_data,
    hamiltonian_type,
):
    """Local pulse - targe qubit 1:
    pulser.Pulse(RampWaveform(5,3,10),RampWaveform(5,1.5,-10),0.1) and
    Global pulse: Pulse(RampWaveform(8,10.0,0.0),RampWaveform(8,-10,10),0.2)"""
    TEST_DURATION = 13
    dt = 2
    target_times = torch.arange(0, TEST_DURATION + 1, dt).tolist()
    sequence.get_duration.return_value = TEST_DURATION

    noisy_samples = mock_sample(hamiltonian_type)
    noisy_samples.max_duration = TEST_DURATION

    actual_omega, actual_delta, actual_phi = _extract_omega_delta_phi(
        noisy_samples=noisy_samples, target_times=target_times, qubit_ids=TEST_QUBIT_IDS
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
        dtype=dtype,
    )
    expected_delta = torch.tensor(
        [
            [-1.3750, 0.0000, 0.0000],
            [-7.1250, 0.0000, 0.0000],
            [-10.0000, -10.0000, -10.0000],
            [-4.2857, -4.2857, -4.2857],
            [1.4286, 1.4286, 1.4286],
            [7.1429, 7.1429, 7.1429],
        ],
        dtype=dtype,
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
        dtype=dtype,
    )
    assert torch.allclose(actual_omega, expected_omega, rtol=0, atol=1e-4)
    assert torch.allclose(actual_delta, expected_delta, rtol=0, atol=1e-4)
    assert torch.allclose(actual_phi, expected_phi, rtol=0, atol=1e-4)


@pytest.mark.parametrize(
    ("hamiltonian_type"),
    [
        "ground-rydberg",
        "XY",
    ],
)
@patch("emu_base.pulser_adapter.HamiltonianData")
def test_extract_omega_delta_phi_dt_1(
    mock_data,
    hamiltonian_type,
):
    """Local pulse - targe qubit 1:
    pulser.Pulse(RampWaveform(5,3,10),RampWaveform(5,1.5,-10),0.1) and
    Global pulse: Pulse(RampWaveform(8,10.0,0.0),RampWaveform(8,-10,10),0.2)"""
    TEST_DURATION = 13
    dt = 1
    target_times = torch.arange(0, TEST_DURATION + 1, dt).tolist()
    sequence.get_duration.return_value = TEST_DURATION

    noisy_samples = mock_sample(hamiltonian_type)
    noisy_samples.max_duration = TEST_DURATION

    actual_omega, actual_delta, actual_phi = _extract_omega_delta_phi(
        noisy_samples=noisy_samples, target_times=target_times, qubit_ids=TEST_QUBIT_IDS
    )

    expected_number_of_samples = math.ceil(TEST_DURATION / dt - 0.5)
    assert len(actual_omega) == expected_number_of_samples

    expected_omega = torch.tensor(
        [
            [
                3.875,
                5.625,
                7.375,
                9.125,
                10.0,
                9.285714285000001,
                7.857142855,
                6.428571425,
                5.0,
                3.5714285749999997,
                2.1428571450000002,
                0.714285715,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                5.0,
                9.285714285000001,
                7.857142855,
                6.428571425,
                5.0,
                3.5714285749999997,
                2.1428571450000002,
                0.714285715,
                0.0,
            ],
            [
                3.875,
                5.625,
                7.375,
                9.125,
                10.0,
                9.285714285000001,
                7.857142855,
                6.428571425,
                5.0,
                3.5714285749999997,
                2.1428571450000002,
                0.714285715,
                0.0,
            ],
        ],
        dtype=dtype,
    ).T
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
        dtype=dtype,
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
        dtype=dtype,
    )

    assert torch.allclose(actual_omega, expected_omega, rtol=0, atol=1e-4)
    assert torch.allclose(actual_delta, expected_delta, rtol=0, atol=1e-4)
    assert torch.allclose(actual_phi, expected_phi, rtol=0, atol=1e-4)


@patch("emu_base.pulser_adapter.HamiltonianData")
def test_autograd(mock_data):
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
        dtype=dtype,
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
        dtype=dtype,
        requires_grad=True,
    )
    phase_tensor = torch.tensor(
        [0.2] * 10,
        dtype=dtype,
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
    mock_from_sequence = MagicMock()
    sample_mock.to_nested_dict.return_value = {"Local": mock_pulser_dict}
    mock_data.from_sequence.return_value = mock_from_sequence
    mock_from_sequence.noisy_samples = sample_mock
    mock_from_sequence.noisy_samples.max_duration = TEST_DURATION
    mock_from_sequence.interaction_type = "ising"

    # first data for grad
    omega_value = _extract_omega_delta_phi(
        noisy_samples=sample_mock, target_times=target_times, qubit_ids=TEST_QUBIT_IDS
    )[0][2, 2].real

    # second data to for grad
    dict_sample = sample_mock.to_nested_dict()
    dict_sample_atom2 = dict_sample["Local"]["ground-rydberg"][TEST_QUBIT_IDS[2]]["amp"]

    res = torch.autograd.grad(
        omega_value,
        dict_sample_atom2,
    )
    expected = torch.zeros(TEST_DURATION, dtype=dtype)
    expected[5] = 1
    assert torch.allclose(res[0], expected)


def test_get_all_lindblad_operators_no_noise():
    noise_model = NoiseModel()

    assert _get_all_lindblad_noise_operators(noise_model) == []


def test_get_all_lindblad_operators():
    random_collapse = torch.rand(2, 2, dtype=dtype)

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
            dtype=dtype,
        ),
    )

    assert torch.allclose(
        ops[2],
        torch.tensor(
            [
                [0, -0.2j],
                [0.2j, 0],
            ],
            dtype=dtype,
        ),
    )

    assert torch.allclose(
        ops[3],
        torch.tensor(
            [
                [0.2, 0],
                [0, -0.2],
            ],
            dtype=dtype,
        ),
    )

    # Dephasing
    assert torch.allclose(
        ops[0],
        torch.tensor(
            [
                [0.05, 0],
                [0, -0.05],
            ],
            dtype=dtype,
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


@patch("emu_base.pulser_adapter.HamiltonianData")
def test_parsed_sequence(mock_data):
    TEST_DURATION = 10
    dt = 2
    adressed_basis = "XY"

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
        dtype=dtype,
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
        dtype=dtype,
    )
    phase_tensor = torch.tensor([0.2] * 10, dtype=dtype)

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
    mock_from_sequence = MagicMock()
    mock_data.from_sequence.return_value = mock_from_sequence
    mock_from_sequence.dim = 2
    mock_from_sequence.noisy_samples = sample_instance
    mock_from_sequence.noisy_samples.max_duration = TEST_DURATION
    mock_from_sequence.interaction_type = adressed_basis

    interaction_matrix = [
        [0.0, 0.0929, -0.4],
        [0.0929, 0.0, 0.1067],
        [-0.4, 0.1067, 0.0],
    ]
    sequence._slm_mask_time = []

    random_collapse = torch.rand(2, 2, dtype=dtype)  # in pulser XY basis
    effective_noise_rates = [0.0036]
    noise_model = NoiseModel(
        depolarizing_rate=0.16,
        dephasing_rate=0.005,
        eff_noise_rates=effective_noise_rates,
        eff_noise_opers=[random_collapse],
    )

    config = EmulationConfig(
        observables=[mock_observable],
        noise_model=noise_model,
        interaction_matrix=interaction_matrix,
        interaction_cutoff=0.15,
    )

    parsed_sequence = PulserData(sequence=sequence, config=config, dt=dt)
    omega, delta, phi = _extract_omega_delta_phi(
        sample_instance, target_times=target_times, qubit_ids=TEST_QUBIT_IDS
    )

    cutoff_interaction_matrix = torch.tensor(
        [[0.0, 0.0, -0.4], [0.0, 0.0, 0.0], [-0.4, 0.0, 0.0]],
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

    ops = _get_all_lindblad_noise_operators(
        noise_model, interact_type=parsed_sequence.hamiltonian.interaction_type
    )
    assert len(parsed_sequence.lindblad_ops) == len(ops)
    for i in range(len(ops)):
        assert torch.allclose(ops[i], parsed_sequence.lindblad_ops[i])

    sequence._slm_mask_time = [1.0, 10.0]
    sequence._slm_mask_targets = [1]
    masked_interaction_matrix = cutoff_interaction_matrix.clone().detach()

    parsed_sequence = PulserData(sequence=sequence, config=config, dt=dt)
    omega, delta, phi = _extract_omega_delta_phi(
        sample_instance, target_times=target_times, qubit_ids=TEST_QUBIT_IDS
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

    assert torch.allclose(  # the position of the elements are the same
        ops[-1], math.sqrt(effective_noise_rates[0]) * random_collapse
    )


@patch("emu_base.pulser_adapter.HamiltonianData")
def test_pulser_data(mock_data):
    TEST_DURATION = 10
    dt = 2

    mock_from_sequence = MagicMock()
    mock_data.from_sequence.return_value = mock_from_sequence
    target_times = torch.arange(0, TEST_DURATION + 1, dt).tolist()
    sequence.get_duration.return_value = TEST_DURATION
    adressed_basis = "ground-rydberg"
    sequence.get_addressed_bases.return_value = [adressed_basis]
    mock_from_sequence.noisy_samples = mock_sample(adressed_basis)
    mock_from_sequence.noisy_samples.max_duration = TEST_DURATION
    mock_from_sequence.interaction_type = "ising"

    mat = torch.randn(3, 3, dtype=float)
    interaction_matrix = (mat + mat.T).fill_diagonal_(0).tolist()

    sequence._slm_mask_time = []

    noise_model = NoiseModel(
        laser_waist=0.1, amp_sigma=1.0, runs=1, samples_per_run=1, temperature=5.0
    )

    config = EmulationConfig(
        observables=[mock_observable],
        noise_model=noise_model,
        interaction_matrix=interaction_matrix,
        interaction_cutoff=0.15,
    )
    torch.manual_seed(1337)
    parsed_sequence = PulserData(sequence=sequence, config=config, dt=dt)
    torch.manual_seed(1337)
    omega, delta, phi = _extract_omega_delta_phi(
        noisy_samples=mock_from_sequence.noisy_samples,
        target_times=target_times,
        qubit_ids=sequence.register.qubit_ids,
    )
    assert torch.allclose(omega, parsed_sequence.omega)
    assert torch.allclose(delta, parsed_sequence.delta)
    assert torch.allclose(phi, parsed_sequence.phi)


@pytest.mark.parametrize("with_modulation", [True, False])
def test_get_target_times_with_obs_eval_time(with_modulation):
    duration = 123
    dt = 3

    eval_times_full = [0, 0.9, 61.5, 110.7]  # just some fractional times
    eval_times = [t / duration for t in eval_times_full]
    obs = MagicMock(spec=Observable, evaluation_times=eval_times)

    config = EmulationConfig(
        observables=[obs], interaction_cutoff=0.0, with_modulation=with_modulation
    )

    with patch.object(sequence, "get_duration") as mock_get_duration:

        mock_get_duration.return_value = duration

        target_times = _get_target_times(sequence, config, dt)

        expected_times = list(range(0, duration + 1, dt))
        expected_times += eval_times_full

        expected = sorted(expected_times)

        assert target_times == expected

        # get_duration called with the correct 'include_fall_time'
        mock_get_duration.assert_called_once_with(include_fall_time=with_modulation)


@pytest.mark.parametrize("prefer_device_model", [True, False])
def test_non_lindbladian_noise(prefer_device_model):
    q_dict = {
        "q0": [-4.0, 0.0],
        "q1": [4.0, 0.0],
    }
    reg = pulser.Register(q_dict)
    seq = pulser.Sequence(reg, pulser.MockDevice)

    seq.declare_channel("ch0", "rydberg_global")

    seq.add(
        pulser.Pulse.ConstantDetuning(
            pulser.BlackmanWaveform(200, torch.pi / 5), 0.0, 0.0
        ),
        "ch0",
    )

    noise = NoiseModel(
        amp_sigma=1.0,
        detuning_sigma=1.0,
        temperature=10.0,
        trap_depth=1.0,
        trap_waist=1.0,
        state_prep_error=0.5,
        runs=1,
    )
    config = EmulationConfig(
        interaction_cutoff=0.0,
        noise_model=noise,
        prefer_device_noise_model=prefer_device_model,
    )

    # this should not error if the custom noise model is used
    data = PulserData(sequence=seq, config=config, dt=10)
    assert data.noise_model == (
        noise if not prefer_device_model else pulser.MockDevice.default_noise_model
    )


def test_extract_omega_delta_phi_missing_qubit():
    """
    Test that qubits not present in the pulse samples are filtered out
    Only qubits included in the pulser sequence are used to fill omega, delta, and phi.
    """
    pulse_duration = 5
    target_times = list(range(pulse_duration + 1))
    qubit_ids = ["q0", "q1", "q2"]

    mock_pulser_dict = {
        "ground-rydberg": {
            "q0": {
                "amp": [1, 2, 3, 4, 5, 6],
                "det": [0, 0, 0, 0, 0, 0],
                "phase": [0, 0, 0, 0, 0, 0],
            },
            "q2": {
                "amp": [6, 5, 4, 3, 2, 1],
                "det": [0, 0, 0, 0, 0, 0],
                "phase": [0, 0, 0, 0, 0, 0],
            },
        }
    }

    sample_mock = MagicMock()
    sample_mock.to_nested_dict.return_value = {"Local": mock_pulser_dict}
    sample_mock.max_duration = pulse_duration

    omega, delta, phi = _extract_omega_delta_phi(
        noisy_samples=sample_mock,
        qubit_ids=qubit_ids,
        target_times=target_times,
    )
    # Check values of omega for q0 and q2
    qubit_map = {0: "q0", 1: "q2"}
    for q_idx, q_id in qubit_map.items():
        for i in range(omega.shape[0]):
            expected_omega = (
                mock_pulser_dict["ground-rydberg"][q_id]["amp"][i]
                + mock_pulser_dict["ground-rydberg"][q_id]["amp"][i + 1]
            ) / 2
            assert omega[i, q_idx] == expected_omega
