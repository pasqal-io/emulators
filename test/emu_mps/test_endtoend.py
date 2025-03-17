import time
from unittest.mock import ANY, MagicMock, patch

import pulser
import pytest
import torch
import random
from pytest import approx

import emu_mps
from emu_mps import (
    MPS,
    BitStrings,
    Fidelity,
    MPSBackend,
    MPSConfig,
    StateResult,
    Occupation,
    Energy,
    EnergyVariance,
    EnergySecondMoment,
    CorrelationMatrix,
)

import pulser.noise_model

from emu_mps.mps_backend_impl import MPSBackendImpl
from test.utils_testing import (
    pulser_afm_sequence_grid,
    pulser_afm_sequence_ring,
    pulser_XY_sequence_slm_mask,
)

seed = 1337


def create_antiferromagnetic_mps(num_qubits: int):
    factors = [torch.zeros((1, 2, 1), dtype=torch.complex128) for _ in range(num_qubits)]
    for i in range(num_qubits):
        if i % 2:
            factors[i][0, 0, 0] = 1.0
        else:
            factors[i][0, 1, 0] = 1.0
    return MPS(factors)


def simulate(
    seq: pulser.Sequence,
    *,
    dt=100,
    noise_model=None,
    state_prep_error=0.0,
    p_false_pos=0.0,
    p_false_neg=0.0,
    initial_state=None,
    given_fidelity_state=True,
    interaction_cutoff=0.0,
):
    if given_fidelity_state:
        fidelity_state = create_antiferromagnetic_mps(len(seq.register.qubit_ids))
    else:
        fidelity_state = MPS.make(len(seq.register.qubit_ids))

    if state_prep_error > 0.0 or p_false_pos > 0.0 or p_false_neg > 0.0:
        assert noise_model is None, "Provide either noise_model or SPAM values"

        runs_args = (
            {"runs": 1, "samples_per_run": 1} if state_prep_error > 0.0 else {}
        )  # Avoid Pulser warning

        noise_model = pulser.noise_model.NoiseModel(
            **runs_args,
            state_prep_error=state_prep_error,
            p_false_pos=p_false_pos,
            p_false_neg=p_false_neg,
        )
    else:
        if noise_model is None:
            noise_model = pulser.noise_model.NoiseModel()
    times = [1.0]
    mps_config = MPSConfig(
        initial_state=initial_state,
        dt=dt,
        precision=1e-5,
        observables=[
            StateResult(evaluation_times=times),
            BitStrings(evaluation_times=times, num_shots=1000),
            Fidelity(evaluation_times=times, state=fidelity_state, tag_suffix="1"),
            Occupation(evaluation_times=times),
            Energy(evaluation_times=times),
            EnergyVariance(evaluation_times=times),
            EnergySecondMoment(evaluation_times=times),
            CorrelationMatrix(evaluation_times=times),
        ],
        noise_model=noise_model,
        interaction_cutoff=interaction_cutoff,
    )

    backend = MPSBackend(seq, config=mps_config)
    result = backend.run()

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
    return simulate(seq, **kwargs)


def get_proba(state: MPS, bitstring: str):
    # FIXME: use MPS factory method from bitstring
    one = torch.tensor([[[0], [1]]], dtype=torch.complex128)
    zero = torch.tensor([[[1], [0]]], dtype=torch.complex128)

    factors = [one if bitstring[i] == "1" else zero for i in range(state.num_sites)]

    return abs(state.inner(MPS(factors))) ** 2


Omega_max = 4 * 2 * torch.pi
U = Omega_max / 2
delta_0 = -6 * U
delta_f = 2 * U
t_rise = 500
t_fall = 1000


def test_XY_3atoms():
    torch.manual_seed(seed)
    seq = pulser_XY_sequence_slm_mask(amplitude=25.0)

    result = simulate(seq, dt=10, given_fidelity_state=False)

    final_state: MPS = result.state[-1]
    final_vec = torch.einsum("abc,cde,efg->abdfg", *(final_state.factors)).reshape(8)

    expected_res = torch.tensor(
        [
            -0.0684 - 0.5677j,
            0.0202 - 0.0305j,
            -0.0313 + 0.0214j,
            -0.2322 + 0.3942j,
            0.0202 - 0.0305j,
            -0.2329 + 0.3709j,
            -0.2322 + 0.3942j,
            0.2344 - 0.0602j,
        ],
        device=final_state.factors[0].device,
        dtype=torch.complex128,
    )

    # pulser magnetization: [0.46024234949993825,0.4776498885102908,0.4602423494999386#
    q_density = result.occupation[-1]

    max_bond_dim = final_state.get_max_bond_dim()
    assert max_bond_dim == 2
    assert approx(q_density.tolist(), 1e-3) == [0.4610, 0.4786, 0.4610]
    assert torch.allclose(final_vec, expected_res, rtol=0, atol=1e-4)


def test_XY_3atomswith_slm():
    torch.manual_seed(seed)
    seq = pulser_XY_sequence_slm_mask(amplitude=0.0, slm_masked_atoms=(1, 2))

    result = simulate(seq, dt=10, given_fidelity_state=False)

    final_state: MPS = result.state[-1]
    final_vec = torch.einsum("abc,cde,efg->abdfg", *(final_state.factors)).reshape(8)
    # pulser vector: 0.707,(−0.171+0.182j),(0.449−0.103j),0.0,(0.138−0.455j),1.761×10 −12,
    # −1.873×10−12j,0.0

    expected_res = torch.tensor(
        [
            7.0711e-01 - 4.2972e-17j,
            -1.7133e-01 + 1.7989e-01j,
            4.4791e-01 - 1.0291e-01j,
            2.2578e-16 - 4.4738e-15j,
            1.3729e-01 - 4.5631e-01j,
            2.1802e-15 - 3.0011e-16j,
            -1.7551e-15 + 2.5736e-15j,
            2.6618e-15 + 5.4529e-16j,
        ],
        device=final_state.factors[0].device,
        dtype=torch.complex128,
    )
    # pulser magnetization: [0.22572457283642877,0.21208108307887844,0.06213666344288577
    q_density = result.occupation[-1]

    max_bond_dim = final_state.get_max_bond_dim()
    assert max_bond_dim == 2
    assert approx(q_density, 1e-3) == [0.2270, 0.2112, 0.0617]
    assert torch.allclose(
        final_vec, expected_res, rtol=0, atol=1e-4
    )  # todo, compare against pulser results


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

    bitstrings = result.bitstrings[-1]
    final_state = result.state[-1]
    final_fidelity = result.fidelity_1[-1]
    max_bond_dim = final_state.get_max_bond_dim()
    fidelity_state = create_antiferromagnetic_mps(num_qubits)

    assert bitstrings["1010101010"] == 129  # -> fidelity as samples increase
    assert bitstrings["0101010101"] == 135
    assert fidelity_state.overlap(final_state) == approx(final_fidelity, abs=1e-10)
    assert max_bond_dim == 29

    q_density = result.occupation[-1]
    assert approx(q_density, 1e-3) == [0.578] * 10

    energy = result.energy[-1]
    assert approx(energy, 1e-8) == -115.34370829396005

    energy_variance = result.energy_variance[-1]
    assert approx(energy_variance, 1e-6) == 45.905980469959104

    second_moment_energy = result.energy_second_moment[-1]
    assert approx(second_moment_energy, 1e-6) == 13350.07680148

    correlation_matrix = result.correlation_matrix[-1]
    print(correlation_matrix)


def test_end_to_end_afm_line_with_state_preparation_errors():
    torch.manual_seed(seed)

    with patch(
        "emu_mps.mps_backend_impl.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        pick_well_prepared_qubits_mock.return_value = [True, True, True, False]
        result = simulate_line(4, state_prep_error=0.1)
        final_state = result.state[-1]
        pick_well_prepared_qubits_mock.assert_called_with(0.1, 4)

    assert get_proba(final_state, "1110") == approx(0.56, abs=1e-2)
    assert get_proba(final_state, "1010") == approx(0.43, abs=1e-2)

    # A dark qubit at the end of the line gives the same result as a line with one less qubit.
    with patch(
        "emu_mps.mps_backend_impl.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        result = simulate_line(3)
        final_state = result.state[-1]
        pick_well_prepared_qubits_mock.assert_not_called()
        assert get_proba(final_state, "111") == approx(0.56, abs=1e-2)
        assert get_proba(final_state, "101") == approx(0.43, abs=1e-2)

    with patch(
        "emu_mps.mps_backend_impl.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        pick_well_prepared_qubits_mock.return_value = [True, False, True, True]
        result = simulate_line(4, state_prep_error=0.1)
        final_state = result.state[-1]

    assert get_proba(final_state, "1011") == approx(0.95, abs=1e-2)

    # Results for a 2 qubit line.
    result = simulate_line(2)
    final_state = result.state[-1]
    assert get_proba(final_state, "11") == approx(0.95, abs=1e-2)

    with patch(
        "emu_mps.mps_backend_impl.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        pick_well_prepared_qubits_mock.return_value = [False, True, True, False]
        result = simulate_line(4, state_prep_error=0.1)
        final_state = result.state[-1]

    assert get_proba(final_state, "0110") == approx(0.95, abs=1e-2)

    # FIXME: When n-1 qubits are dark, the simulation fails!
    with patch(
        "emu_mps.mps_backend_impl.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        with pytest.raises(ValueError) as exception_info:
            pick_well_prepared_qubits_mock.return_value = [False, False, True, False]
            result = simulate_line(4, state_prep_error=0.1)
            final_state = result.state[-1]

    assert "For 1 qubit states, do state vector" in str(exception_info.value)


def test_end_to_end_afm_line_with_measurement_errors():
    with patch("emu_mps.mps.apply_measurement_errors") as apply_measurement_errors_mock:
        bitstrings = MagicMock()
        apply_measurement_errors_mock.return_value = bitstrings
        results = simulate_line(4, p_false_pos=0.0, p_false_neg=0.5)
        apply_measurement_errors_mock.assert_called_with(
            ANY, p_false_pos=0.0, p_false_neg=0.5
        )
        assert results.bitstrings[-1] is bitstrings


def test_initial_state():
    pulse = pulser.Pulse.ConstantAmplitude(
        0.0, pulser.waveforms.ConstantWaveform(10.0, 0.0), 0.0
    )
    reg = pulser.Register.rectangle(5, 1, spacing=1e10, prefix="q")
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")  # do nothing in the pulse

    state = emu_mps.MPS.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"rrrrr": 1.0}
    )
    assert state.inner(state).real == approx(1.0)  # assert unit norm

    state_result = emu_mps.StateResult(evaluation_times=[1.0])
    config = emu_mps.MPSConfig(observables=[state_result], initial_state=state)
    backend = emu_mps.MPSBackend(seq, config=config)
    results = backend.run()
    # assert that the initial state was used by the emulator
    assert results.get_result(state_result, 1.0).inner(state).real == approx(1.0)


def test_initial_state_copy():
    duration = 10.0
    pulse = pulser.Pulse.ConstantAmplitude(
        Omega_max, pulser.waveforms.RampWaveform(duration, delta_0, delta_f), 0.0
    )
    reg = pulser.Register.rectangle(5, 1, spacing=1e10, prefix="q")
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")

    initial_state = emu_mps.MPS.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"rrrrr": 1.0}
    )

    config = emu_mps.MPSConfig(initial_state=initial_state)

    emu_mps.MPSBackend(seq, config=config).run()

    # Check the initial state's factors were not modified.
    assert all(
        torch.allclose(initial_state_factor, expected_initial_state_factor)
        for initial_state_factor, expected_initial_state_factor in zip(
            initial_state.factors,
            emu_mps.MPS.from_state_amplitudes(
                eigenstates=("r", "g"), amplitudes={"rrrrr": 1.0}
            ).factors,
        )
    )


def test_end_to_end_afm_ring_with_noise():
    torch.manual_seed(seed)
    random.seed(0xDEADBEEF)

    num_qubits = 6
    seq = pulser_afm_sequence_ring(
        num_qubits=num_qubits,
        Omega_max=Omega_max,
        U=U,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )

    noise_model = pulser.noise_model.NoiseModel(
        depolarizing_rate=0.1,
    )

    result = simulate(seq, noise_model=noise_model)

    bitstrings = result.bitstrings[-1]
    final_state = result.state[-1]
    max_bond_dim = final_state.get_max_bond_dim()

    assert bitstrings["101010"] == 472
    assert bitstrings["010101"] == 510
    assert max_bond_dim == 8


def test_end_to_end_spontaneous_emission():
    torch.manual_seed(seed)
    random.seed(0xDEADBEEF)

    # Sequence with no driving.
    duration = 10000
    rows, cols = 3, 4
    reg = pulser.Register.rectangle(
        rows, cols, pulser.devices.MockDevice.rydberg_blockade_radius(U), prefix="q"
    )
    seq = pulser.Sequence(reg, pulser.devices.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(
        pulser.Pulse.ConstantAmplitude(
            amplitude=0.0,
            detuning=pulser.waveforms.ConstantWaveform(duration=duration, value=0.0),
            phase=0.0,
        ),
        "ising_global",
    )

    noise_model = pulser.noise_model.NoiseModel(
        relaxation_rate=0.1,
    )

    initial_state = emu_mps.MPS.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"rrrrrrrrrrrr": 1.0}
    )
    result = simulate(seq, noise_model=noise_model, initial_state=initial_state)

    final_state = result.state[-1]

    assert get_proba(final_state, "100000110000") == approx(1, abs=1e-2)

    # Aggregating results of many runs to check the exponential decrease of qubit density
    # would be too much for this unit test.


def test_laser_waist():
    duration = 1000
    reg = pulser.Register.from_coordinates(
        [(0.0, 0.0), (10.0, 0.0)], center=False, prefix="q"
    )
    seq = pulser.Sequence(reg, pulser.devices.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.declare_channel("ising_local", "rydberg_local")
    seq.add(
        pulser.Pulse.ConstantAmplitude(
            amplitude=torch.pi,
            detuning=pulser.waveforms.ConstantWaveform(duration=duration, value=0.0),
            phase=0.0,
        ),
        "ising_global",
    )
    e_inv = 0.36787944117144233
    seq.target(seq.register.qubit_ids[1], "ising_local")
    seq.add(
        pulser.Pulse.ConstantAmplitude(
            amplitude=torch.pi * (1 - e_inv),
            detuning=pulser.waveforms.ConstantWaveform(duration=duration, value=0.0),
            phase=0.0,
        ),
        "ising_local",
    )

    noise_model = pulser.noise_model.NoiseModel(
        laser_waist=10.0,
    )

    result = simulate(seq, noise_model=noise_model, dt=10, interaction_cutoff=100)

    final_state = result.state[-1]

    assert pytest.approx(final_state.norm()) == 1.0

    expected_state = emu_mps.MPS.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"rr": 1.0}
    )

    assert pytest.approx(final_state.inner(expected_state)) == -1.0


def test_autosave():
    duration = 300
    rows, cols = 2, 3
    reg = pulser.Register.rectangle(
        rows, cols, pulser.devices.MockDevice.rydberg_blockade_radius(U), prefix="q"
    )
    seq = pulser.Sequence(reg, pulser.devices.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(
        pulser.Pulse.ConstantAmplitude(
            amplitude=torch.pi,
            detuning=pulser.waveforms.ConstantWaveform(duration=duration, value=0.0),
            phase=0.0,
        ),
        "ising_global",
    )

    evaluation_times = [1.0 / 30.0, 1.0 / 3.0, 0.5]
    energy = Energy(evaluation_times=evaluation_times)

    save_simulation_original = MPSBackendImpl.save_simulation
    save_file = None

    counter = 100  # Number of simulation steps before crashing

    def save_simulation_mock_side_effect(self):
        nonlocal counter
        counter -= 1
        if counter > 0:
            self.last_save_time = time.time() + 999
            return save_simulation_original(self)

        assert self.timestep_index == 11

        self.last_save_time = 0  # Trigger saving regardless of time
        save_simulation_original(self)
        nonlocal save_file
        save_file = self.autosave_file
        raise Exception("Process killed!")

    with patch.object(
        MPSBackendImpl, "save_simulation", autospec=True
    ) as save_simulation_mock:
        save_simulation_mock.side_effect = save_simulation_mock_side_effect

        with pytest.raises(Exception) as e:
            MPSBackend(seq, config=MPSConfig(observables=[energy])).run()

        assert str(e.value) == "Process killed!"

    assert save_file is not None and save_file.is_file()
    results_after_resume = MPSBackend.resume(save_file)

    assert not save_file.is_file()

    results_expected = MPSBackend(seq, MPSConfig(observables=[energy])).run()

    for t in evaluation_times:
        assert results_after_resume["energy", t] == results_expected["energy", t]
