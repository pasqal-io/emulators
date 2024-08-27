from unittest.mock import ANY, MagicMock, patch

import pulser
import pytest
import torch
from pytest import approx

import emu_mps
import emu_mps.base_classes
import emu_mps.base_classes.default_callbacks
from emu_mps import MPS, BitStrings, Fidelity, MPSBackend, MPSConfig, StateResult

from .utils_testing import pulser_afm_sequence_grid, pulser_afm_sequence_ring


seed = 1337

mps_backend = MPSBackend()


def create_antiferromagnetic_mps(num_qubits: int):
    factors = [torch.zeros((1, 2, 1), dtype=torch.complex128) for _ in range(num_qubits)]
    for i in range(num_qubits):
        if i % 2:
            factors[i][0, 0, 0] = 1.0
        else:
            factors[i][0, 1, 0] = 1.0
    return MPS(factors)


def simulate(seq, state_prep_error=0.0, p_false_pos=0.0, p_false_neg=0.0):
    final_time = seq.get_duration()
    fidelity_state = create_antiferromagnetic_mps(len(seq.register.qubit_ids))

    noise_model = None
    if state_prep_error > 0.0 or p_false_pos > 0.0 or p_false_neg > 0.0:
        noise_model = pulser.noise_model.NoiseModel(
            noise_types=("SPAM",),
            state_prep_error=state_prep_error,
            p_false_pos=p_false_pos,
            p_false_neg=p_false_neg,
        )

    times = {final_time}
    mps_config = MPSConfig(
        dt=100,
        precision=1e-5,
        observables=[
            StateResult(evaluation_times=times),
            BitStrings(evaluation_times=times, num_shots=1000),
            Fidelity(evaluation_times=times, state=fidelity_state),
        ],
        noise_model=noise_model,
    )

    result = mps_backend.run(seq, mps_config)

    return result


def simulate_line(n, **kwargs):
    seq = pulser_afm_sequence_grid(
        rows=1,
        columns=n,
        Omega_max=Omega_max,
        U=U,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )
    return seq.get_duration(), simulate(seq, **kwargs)


def get_proba(state: MPS, bitstring: str):
    # FIXME: use MPS factory method from bitstring
    one = torch.tensor([[[0], [1]]], dtype=torch.complex128)
    zero = torch.tensor([[[1], [0]]], dtype=torch.complex128)

    factors = [one if bitstring[i] == "1" else zero for i in range(state.num_sites)]

    return abs(state.inner(MPS(factors, truncate=False))) ** 2


Omega_max = 4 * 2 * torch.pi
U = Omega_max / 2
delta_0 = -6 * U
delta_f = 2 * U
t_rise = 500
t_fall = 1000


def test_end_to_end_afm_ring():
    torch.manual_seed(seed)

    num_qubits = 10
    seq = pulser_afm_sequence_ring(
        num_qubits=num_qubits,
        Omega_max=Omega_max,
        U=U,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )

    result = simulate(seq)

    final_time = seq.get_duration()
    bitstrings = result["bitstrings"][final_time]
    final_state = result["state"][final_time]
    final_fidelity = result[
        f"fidelity_{emu_mps.base_classes.default_callbacks._fidelity_counter}"
    ][final_time]
    max_bond_dim = final_state.get_max_bond_dim()
    fidelity_state = create_antiferromagnetic_mps(num_qubits)

    assert bitstrings["1010101010"] == 129  # -> fidelity as samples increase
    assert bitstrings["0101010101"] == 135
    assert fidelity_state.inner(final_state) == approx(final_fidelity, abs=1e-10)
    assert max_bond_dim == 29


def test_end_to_end_afm_line_with_state_preparation_errors():
    torch.manual_seed(seed)

    with patch(
        "emu_mps.mps_backend.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        pick_well_prepared_qubits_mock.return_value = [True, True, True, False]
        final_time, result = simulate_line(4, state_prep_error=0.1)
        final_state = result["state"][final_time]
        pick_well_prepared_qubits_mock.assert_called_with(0.1, 4)

    assert get_proba(final_state, "1110") == approx(0.56, abs=1e-2)
    assert get_proba(final_state, "1010") == approx(0.43, abs=1e-2)

    # A dark qubit at the end of the line gives the same result as a line with one less qubit.
    with patch(
        "emu_mps.mps_backend.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        final_time, result = simulate_line(3)
        final_state = result["state"][final_time]
        pick_well_prepared_qubits_mock.assert_not_called()
        assert get_proba(final_state, "111") == approx(0.56, abs=1e-2)
        assert get_proba(final_state, "101") == approx(0.43, abs=1e-2)

    with patch(
        "emu_mps.mps_backend.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        pick_well_prepared_qubits_mock.return_value = [True, False, True, True]
        final_time, result = simulate_line(4, state_prep_error=0.1)
        final_state = result["state"][final_time]

    assert get_proba(final_state, "1011") == approx(0.95, abs=1e-2)

    # Results for a 2 qubit line.
    final_time, result = simulate_line(2)
    final_state = result["state"][final_time]
    assert get_proba(final_state, "11") == approx(0.95, abs=1e-2)

    with patch(
        "emu_mps.mps_backend.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        pick_well_prepared_qubits_mock.return_value = [False, True, True, False]
        final_time, result = simulate_line(4, state_prep_error=0.1)
        final_state = result["state"][final_time]

    assert get_proba(final_state, "0110") == approx(0.95, abs=1e-2)

    # FIXME: When n-1 qubits are dark, the simulation fails!
    with patch(
        "emu_mps.mps_backend.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        with pytest.raises(ValueError) as exception_info:
            pick_well_prepared_qubits_mock.return_value = [False, False, True, False]
            final_time, result = simulate_line(4, state_prep_error=0.1)
            final_state = result["state"][final_time]

    assert "For 1 qubit states, do state vector" in str(exception_info.value)


def test_end_to_end_afm_line_with_measurement_errors():
    with patch("emu_mps.mps.apply_measurement_errors") as apply_measurement_errors_mock:
        bitstrings = MagicMock()
        apply_measurement_errors_mock.return_value = bitstrings
        final_time, results = simulate_line(4, p_false_pos=0.0, p_false_neg=0.5)
        apply_measurement_errors_mock.assert_called_with(
            ANY, p_false_pos=0.0, p_false_neg=0.5
        )
        assert results["bitstrings"][final_time] is bitstrings


def test_initial_state():
    pulse = pulser.Pulse.ConstantAmplitude(
        0.0, pulser.waveforms.ConstantWaveform(10.0, 0.0), 0.0
    )
    reg = pulser.Register.rectangle(5, 1, spacing=1e10)
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")  # do nothing in the pulse

    state = emu_mps.MPS.from_state_string(
        basis=("r", "g"), nqubits=5, strings={"rrrrr": 1.0}
    )
    assert state.inner(state).real == approx(1.0)  # assert unit norm

    state_result = emu_mps.StateResult(evaluation_times={10})
    config = emu_mps.MPSConfig(observables=[state_result], initial_state=state)
    backend = emu_mps.MPSBackend()
    results = backend.run(seq, config)
    # assert that the initial state was used by the emulator
    assert results[state_result.name()][10].inner(state).real == approx(1.0)
