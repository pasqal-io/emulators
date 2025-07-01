import math
import random
from unittest.mock import ANY, MagicMock, patch
import pytest
import torch
from pytest import approx
from typing import Any

import pulser
import pulser.noise_model

from emu_sv import (
    SVBackend,
    SVConfig,
    StateVector,
    CorrelationMatrix,
    EnergySecondMoment,
    EnergyVariance,
    Occupation,
    BitStrings,
    Energy,
    Fidelity,
    Results,
    StateResult,
    DensityMatrix,
)

from test.utils_testing import (
    pulser_afm_sequence_ring,
    pulser_blackman,
)


seed = 1337


Omega_max = 4 * 2 * torch.pi
U = Omega_max / 2
delta_0 = -6 * U
delta_f = 2 * U
t_rise = 500
t_fall = 1000

dtype = torch.complex128
dtype_f64 = torch.float64
gpu = False  # if True, bitstrigs and energy variance will fail

device = "cpu" if not gpu else "cuda"


def create_antiferromagnetic_state_vector(
    num_qubits: int, gpu: bool = True
) -> StateVector:
    factors = [torch.zeros(2, dtype=dtype) for _ in range(num_qubits)]
    for i in range(num_qubits):
        if i % 2:
            factors[i][0] = 1.0
        else:
            factors[i][1] = 1.0

    afm_vec = factors[0]
    for i in factors[1:]:
        afm_vec = torch.kron(afm_vec, i)

    return StateVector(afm_vec, gpu=gpu)


def simulate(
    seq: pulser.Sequence,
    *,
    dt: int = 100,
    noise_model: Any | None = None,
    state_prep_error: float = 0,
    p_false_pos: float = 0,
    p_false_neg: float = 0,
    initial_state: Any | None = None,
    given_fidelity_state: bool = True,
    interaction_cutoff: float = 0,
    gpu: bool = True,
    num_shots: int = 1000,
) -> Results:
    n_qubits = len(seq.register.qubit_ids)
    if noise_model is None:
        noise_model = pulser.noise_model.NoiseModel()

    if given_fidelity_state:
        fidelity_state = create_antiferromagnetic_state_vector(n_qubits, gpu=gpu)
    else:
        fidelity_state = StateVector.make(n_qubits, gpu=gpu)  # make |00.0>

    if state_prep_error > 0.0 or p_false_pos > 0.0 or p_false_neg > 0.0:

        runs_args = (
            {"runs": 1, "samples_per_run": 1} if state_prep_error > 0.0 else {}
        )  # Avoid Pulser warning

        noise_model = pulser.noise_model.NoiseModel(
            **runs_args,
            state_prep_error=state_prep_error,
            p_false_pos=p_false_pos,
            p_false_neg=p_false_neg,
        )

    times = [1.0]
    sv_config = SVConfig(
        initial_state=initial_state,
        dt=dt,
        krylov_tolerance=1e-5,
        observables=[
            StateResult(evaluation_times=times),
            BitStrings(evaluation_times=times, num_shots=num_shots),
            Fidelity(evaluation_times=times, state=fidelity_state, tag_suffix="1"),
            Occupation(evaluation_times=times),
            Energy(evaluation_times=times),
            EnergyVariance(evaluation_times=times),
            EnergySecondMoment(evaluation_times=times),
            CorrelationMatrix(evaluation_times=times),  # TODO: add a test for this
        ],
        noise_model=noise_model,
        gpu=gpu,
        interaction_cutoff=interaction_cutoff,
    )

    backend = SVBackend(seq, config=sv_config)
    result = backend.run()
    return result


def simulate_with_den_matrix(
    seq: pulser.Sequence,
    *,
    dt: int = 100,
    noise_model: Any,
    state_prep_error: float = 0,
    p_false_pos: float = 0,
    p_false_neg: float = 0,
    initial_state: Any | None = None,
    given_fidelity_state: bool = True,
    interaction_cutoff: float = 0,
    gpu: bool = True,
    num_shots: int = 1000,
) -> Results:
    n_qubits = len(seq.register.qubit_ids)

    if given_fidelity_state:
        fidelity_state = DensityMatrix.from_state_vector(
            create_antiferromagnetic_state_vector(n_qubits, gpu=gpu)
        )
    else:
        fidelity_state = DensityMatrix.make(n_qubits, gpu=gpu)  # make |00.0>

    if state_prep_error > 0.0 or p_false_pos > 0.0 or p_false_neg > 0.0:
        assert noise_model is not None, "Provide either noise_model or SPAM values"

    times = [1.0]
    sv_config = SVConfig(
        initial_state=initial_state,
        dt=dt,
        krylov_tolerance=1e-5,
        observables=[
            StateResult(evaluation_times=times),
            BitStrings(evaluation_times=times, num_shots=num_shots),
            Fidelity(evaluation_times=times, state=fidelity_state, tag_suffix="1"),
            Occupation(evaluation_times=times),
            Energy(evaluation_times=times),
            CorrelationMatrix(evaluation_times=times),  # TODO: add a test for this
            EnergySecondMoment(evaluation_times=times),
            EnergyVariance(evaluation_times=times),
        ],
        noise_model=noise_model,
        gpu=gpu,
        interaction_cutoff=interaction_cutoff,
    )

    backend = SVBackend(seq, config=sv_config)
    result = backend.run()
    return result


def test_end_to_end_afm_ring() -> None:
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

    result = simulate(
        seq, gpu=gpu
    )  # only run on cpu, bitstring sampling is device dependent

    final_time = -1  # seq.get_duration()
    bitstrings = result.bitstrings[final_time]
    final_state = result.state[final_time]
    final_fidelity = result.fidelity_1[final_time]

    fidelity_state = create_antiferromagnetic_state_vector(num_qubits, gpu=gpu)

    assert bitstrings["10" * int(num_qubits / 2)] == 136
    assert bitstrings["01" * int(num_qubits / 2)] == 159
    assert torch.allclose(fidelity_state.overlap(final_state), final_fidelity, atol=1e-10)

    occupation = result.occupation[final_time]

    assert torch.allclose(
        torch.tensor([0.578] * num_qubits, dtype=dtype_f64), occupation, atol=1e-3
    )

    energy = result.energy[final_time]  # (-115.34554274708604-2.1316282072803006e-14j)
    assert torch.allclose(
        energy, torch.tensor(-115.34558020797967, device=device, dtype=dtype_f64)
    )

    energy_variance = result.energy_variance[final_time]  # 45.911110563993134
    assert torch.allclose(
        energy_variance,
        torch.tensor(45.91111056399, dtype=dtype_f64, device=device),
        rtol=1e-3,
    )

    energy_second_moment = result.energy_second_moment[final_time]  # 13350.505342183847
    assert torch.allclose(
        energy_second_moment,
        torch.tensor(13350.5053421, dtype=dtype_f64, device=device),
    )


def test_end_to_end_afm_ring_with_effective_noise() -> None:
    torch.manual_seed(seed)

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
        dephasing_rate=0.1,
    )

    result = simulate_with_den_matrix(
        seq, noise_model=noise_model, gpu=gpu
    )  # only run on cpu, bitstring sampling is device dependent

    final_time = -1  # seq.get_duration()
    bitstrings = result.bitstrings[final_time]
    final_state = result.state[final_time]
    final_fidelity = result.fidelity_1[final_time]

    fidelity_state = DensityMatrix.from_state_vector(
        create_antiferromagnetic_state_vector(num_qubits, gpu=gpu)
    )
    assert bitstrings["10" * int(num_qubits / 2)] == 173
    assert bitstrings["01" * int(num_qubits / 2)] == 168

    assert torch.allclose(fidelity_state.overlap(final_state), final_fidelity, atol=1e-10)

    occupation = result.occupation[final_time]

    assert torch.allclose(
        torch.tensor([0.4596] * num_qubits, dtype=dtype_f64), occupation, atol=1e-3
    )
    energy = result.energy[-1]

    assert torch.allclose(energy, torch.tensor(-53.4424, dtype=dtype_f64, device=device))

    energy_variance = result.energy_variance[final_time]

    assert torch.allclose(
        energy_variance,
        torch.tensor(267.6186, dtype=torch.float64, device="cpu"),
        rtol=1e-3,
    )

    energy_second_moment = result.energy_second_moment[final_time]

    assert torch.allclose(
        energy_second_moment,
        torch.tensor(3123.7050, dtype=torch.float64, device="cpu"),
    )


def test_end_to_end_pi_half_pulse() -> None:
    # π/2 pulse Blackman creates |ψ❭=(|0❭-1j|1❭)/sqrt(2)
    duration = 1000  # ns, so 1 μs
    area = math.pi / 2
    seq = pulser_blackman(duration, area)
    result = simulate(seq, dt=50)

    final_time = -1
    final_state = result.state[final_time]

    expected = torch.tensor([1, -1j], dtype=dtype) / math.sqrt(2)
    assert torch.allclose(final_state.vector.cpu(), expected, atol=1e-8)


def test_end_to_end_pi_half_pulse_with_phase() -> None:
    # π/2 pulse Blackman creates |ψ❭=(|0❭-1j|1❭)/sqrt(2)
    # with a phase factor exp(-1j*duration*φ) = 1j
    duration = 1000  # ns, so 1 μs
    area = math.pi / 2
    phase = math.pi / 2
    seq = pulser_blackman(duration, area, phase=phase)
    result = simulate(seq, dt=50)
    final_time = -1

    final_state = result.state[final_time]
    # with the phase we expect |ψ❭=(|0❭+|1❭)/sqrt(2)
    expected = torch.tensor([1, 1], dtype=dtype) / math.sqrt(2)

    assert torch.allclose(final_state.vector.cpu(), expected, atol=1e-8)


def test_initial_state() -> None:
    pulse = pulser.Pulse.ConstantAmplitude(
        0.0, pulser.waveforms.ConstantWaveform(10.0, 0.0), 0.0
    )
    natoms = 2
    reg = pulser.Register.rectangle(natoms, 1, spacing=1e10, prefix="q")
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")  # do nothing in the pulse

    state = StateVector.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"r" * (natoms): 1.0}
    )
    assert state.norm() == approx(1.0)  # assert unit norm

    state_result = StateResult(evaluation_times=[1.0])
    config = SVConfig(observables=[state_result], initial_state=state)
    backend = SVBackend(seq, config=config)
    results = backend.run()
    # assert that the initial state was used by the emulator
    assert results.get_result(state_result, 1.0).inner(state).real == approx(1.0)
    # but that it's a copy
    assert results.get_result(state_result, 1.0) is not state


def test_initial_state_wrong_atmos_number() -> None:
    pulse = pulser.Pulse.ConstantAmplitude(
        0.0, pulser.waveforms.ConstantWaveform(10.0, 0.0), 0.0
    )
    natoms = 2
    reg = pulser.Register.rectangle(natoms, 1, spacing=1e10, prefix="q")
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")  # do nothing in the pulse

    with pytest.raises(ValueError) as excinfo:
        state = StateVector.from_state_amplitudes(
            eigenstates=("r", "g"), amplitudes={"r" * (natoms + 1): 1.0}
        )
        state_result = StateResult(evaluation_times=[1.0])
        config = SVConfig(observables=[state_result], initial_state=state)
        backend = SVBackend(seq, config=config)
        _ = backend.run()
    assert (
        "Mismatch in number of atoms: initial state has "
        + f"{state.n_qudits} and the sequence has {natoms}"
    ) in str(excinfo.value)


def test_initial_state_with_den_matrix() -> None:
    pulse = pulser.Pulse.ConstantAmplitude(
        0.0, pulser.waveforms.ConstantWaveform(10.0, 0.0), 0.0
    )
    natoms = 5
    reg = pulser.Register.rectangle(natoms, 1, spacing=1e10, prefix="q")
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")  # do nothing in the pulse

    state = DensityMatrix.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"r" * (natoms): 1.0}
    )
    state.matrix = state.matrix.to("cpu")
    assert state.matrix.trace() == approx(1.0)  # assert unit norm

    state_result = StateResult(evaluation_times=[1.0])
    noise_model = pulser.noise_model.NoiseModel(
        dephasing_rate=0.5,
    )
    config = SVConfig(
        observables=[state_result],
        initial_state=state,
        noise_model=noise_model,
        gpu=gpu,
    )
    backend = SVBackend(seq, config=config)
    results = backend.run()
    # assert that the initial state was used by the emulator

    assert torch.allclose(
        results.get_result(state_result, 1.0).matrix.cpu(), state.matrix.cpu(), atol=1e-8
    )

    assert results.get_result(state_result, 1.0) is not state


def test_initial_state_den_mat_wrong_atoms():
    pulse = pulser.Pulse.ConstantAmplitude(
        0.0, pulser.waveforms.ConstantWaveform(10.0, 0.0), 0.0
    )
    natoms = 5
    reg = pulser.Register.rectangle(natoms, 1, spacing=1e10, prefix="q")
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")  # do nothing in the pulse

    state = DensityMatrix.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"r" * (natoms + 1): 1.0}
    )

    with pytest.raises(ValueError) as excinfo:
        state.matrix = state.matrix.to("cpu")

        state_result = StateResult(evaluation_times=[1.0])
        noise_model = pulser.noise_model.NoiseModel(
            dephasing_rate=0.5,
        )
        config = SVConfig(
            observables=[state_result],
            initial_state=state,
            noise_model=noise_model,
            gpu=gpu,
        )
        backend = SVBackend(seq, config=config)
        _ = backend.run()

    assert (
        "Mismatch in number of atoms: initial state has "
        + f"{state.n_qudits} and the sequence has {natoms}"
    ) in str(excinfo.value)


def test_end_to_end_spontaneous_emission_rate() -> None:
    # sequence with spontaneous emission
    seed = 31415
    torch.manual_seed(seed)
    random.seed(0xDEADBEEF)

    total_time = 10000
    pulse = pulser.Pulse.ConstantAmplitude(
        0.0, pulser.waveforms.ConstantWaveform(total_time, 0.0), 0.0
    )
    natoms = 2
    reg = pulser.Register.rectangle(1, natoms, spacing=1e10, prefix="q")
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")

    # emu-sv parameters
    dt = 100
    times = [1.0]
    eigenstate = ("r", "g")
    amplitudes = {"r" * natoms: 1.0}
    relaxation_rate = 0.1

    noise_model = pulser.noise_model.NoiseModel(relaxation_rate=relaxation_rate)
    initial_state = DensityMatrix.from_state_vector(
        StateVector.from_state_amplitudes(eigenstates=eigenstate, amplitudes=amplitudes)
    )
    sv_config = SVConfig(
        initial_state=initial_state,
        dt=dt,
        krylov_tolerance=1e-5,
        observables=[
            StateResult(evaluation_times=times),
            BitStrings(evaluation_times=times, num_shots=1000),
            Occupation(evaluation_times=times),
        ],
        noise_model=noise_model,
        gpu=gpu,
    )
    backend = SVBackend(seq, config=sv_config)
    result = backend.run()

    # pulser results [0.3678794421106907, 0.3678794421106907]
    expected_result = torch.tensor([0.3678] * natoms, dtype=torch.float64)

    assert torch.allclose(result.occupation[-1], expected_result, atol=1e-4)

    expected_counts = {"00": 380, "10": 223, "01": 246, "11": 151}
    assert expected_counts == result.bitstrings[-1]

    # pulser has similar results except upto a basis change
    # diagonal elements: array([0.13533544, 0.232544  , 0.232544  , 0.39957656])
    expected_state = torch.tensor(
        [
            [0.3996 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [0.0000 + 0.0j, 0.2325 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j],
            [0.0000 + 0.0j, 0.0000 + 0.0j, 0.2325 + 0.0j, 0.0000 + 0.0j],
            [0.0000 + 0.0j, 0.0000 + 0.0j, 0.0000 + 0.0j, 0.1353 + 0.0j],
        ],
        dtype=dtype,
        device=device,
    )
    assert torch.allclose(result.state[-1].matrix, expected_state, atol=1e-4)


def test_end_to_end_1D_sv_measure_errors() -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    num_qubits = 4

    # initial state with all atoms in |g> state
    reg = pulser.Register.rectangle(num_qubits, 1, spacing=10, prefix="q")
    pulse = pulser.Pulse.ConstantPulse(1000, 0.0, 0.0, 0.0)

    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")
    # false positive readout error
    result = simulate(
        seq,
        gpu=gpu,
        p_false_neg=0.0,
        p_false_pos=0.1,
        num_shots=10000,
        given_fidelity_state=False,
    )  # only run on cpu, bitstring sampling is device dependent

    final_time = -1  # seq.get_duration()
    bitstrings = result.bitstrings[final_time]

    assert bitstrings["1" * int(num_qubits)] == 1
    assert bitstrings["0" * int(num_qubits)] == 6534
    assert bitstrings["1000"] == 747
    assert bitstrings["0100"] == 719
    assert bitstrings["0010"] == 734
    assert bitstrings["0001"] == 732

    # initial state with all atoms in |r> state
    eigenstate = ("r", "g")
    amplitudes = {"r" * num_qubits: 1.0}
    initial_state = StateVector.from_state_amplitudes(
        eigenstates=eigenstate, amplitudes=amplitudes
    )
    # false negative readout error
    result = simulate(
        seq,
        gpu=gpu,
        initial_state=initial_state,
        p_false_neg=0.1,
        p_false_pos=0.0,
        num_shots=10000,
        given_fidelity_state=False,
    )

    final_time = -1
    bitstrings = result.bitstrings[final_time]

    assert bitstrings["1" * int(num_qubits)] == 6589
    assert bitstrings["0" * int(num_qubits)] == 1
    assert bitstrings["1110"] == 724
    assert bitstrings["1101"] == 766
    assert bitstrings["1011"] == 725
    assert bitstrings["0111"] == 725


def test_end_to_end_1D_dense_mat_measure_errors() -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    num_qubits = 4

    # initial state with all atoms in |g> state
    reg = pulser.Register.rectangle(num_qubits, 1, spacing=10, prefix="q")
    pulse = pulser.Pulse.ConstantPulse(1000, 0.0, 0.0, 0.0)

    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")
    # false positive readout error

    # we should add an effective noise model to the sequence in order to use the
    # density matrix solver
    noise_model = pulser.noise_model.NoiseModel(
        dephasing_rate=0.1, p_false_neg=0.0, p_false_pos=0.1
    )
    result = simulate_with_den_matrix(
        seq,
        gpu=gpu,
        num_shots=10000,
        given_fidelity_state=False,
        noise_model=noise_model,
    )  # only run on cpu, bitstring sampling is device dependent

    final_time = -1  # seq.get_duration()
    bitstrings = result.bitstrings[final_time]

    assert bitstrings["1" * int(num_qubits)] == 1
    assert bitstrings["0" * int(num_qubits)] == 6534
    assert bitstrings["1000"] == 747
    assert bitstrings["0100"] == 719
    assert bitstrings["0010"] == 734
    assert bitstrings["0001"] == 732

    # initial state with all atoms in |r> state
    eigenstate = ("r", "g")
    amplitudes = {"r" * num_qubits: 1.0}
    initial_state = DensityMatrix.from_state_amplitudes(
        eigenstates=eigenstate, amplitudes=amplitudes
    )
    # false negative readout error
    noise_model = pulser.noise_model.NoiseModel(
        dephasing_rate=0.1, p_false_neg=0.1, p_false_pos=0.0
    )
    result = simulate_with_den_matrix(
        seq,
        noise_model=noise_model,
        gpu=gpu,
        initial_state=initial_state,
        num_shots=10000,
        given_fidelity_state=False,
    )

    final_time = -1
    bitstrings = result.bitstrings[final_time]
    print(bitstrings)
    assert bitstrings["1" * int(num_qubits)] == 6589
    assert bitstrings["0" * int(num_qubits)] == 1
    assert bitstrings["1110"] == 724
    assert bitstrings["1101"] == 766
    assert bitstrings["1011"] == 725
    assert bitstrings["0111"] == 725


def test_spam_bitstring_class() -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    num_qubits = 4

    reg = pulser.Register.rectangle(num_qubits, 1, spacing=10, prefix="q")
    pulse = pulser.Pulse.ConstantPulse(1000, 0.0, 0.0, 0.0)

    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")
    with patch(
        "emu_sv.state_vector.apply_measurement_errors"
    ) as apply_measurement_errors_mock:
        bitstrings = MagicMock()
        apply_measurement_errors_mock.return_value = bitstrings

        results = simulate(seq, p_false_pos=0.0, p_false_neg=0.5)
        apply_measurement_errors_mock.assert_called_with(
            ANY, p_false_pos=0.0, p_false_neg=0.5
        )
        assert results.bitstrings[-1] is bitstrings
    with patch(
        "emu_sv.density_matrix_state.apply_measurement_errors"
    ) as apply_measurement_errors_mock:
        bitstrings = MagicMock()
        apply_measurement_errors_mock.return_value = bitstrings
        noise_model = pulser.noise_model.NoiseModel(
            dephasing_rate=0.1, p_false_neg=0.5, p_false_pos=0.0
        )
        results = simulate_with_den_matrix(
            seq, p_false_pos=0.0, p_false_neg=0.5, noise_model=noise_model
        )
        apply_measurement_errors_mock.assert_called_with(
            ANY, p_false_pos=0.0, p_false_neg=0.5
        )
        assert results.bitstrings[-1] is bitstrings
