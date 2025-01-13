from pytest import approx
import pulser
import torch

import emu_base.base_classes
import emu_base.base_classes.default_callbacks

from emu_base.base_classes import BitStrings, Fidelity, StateResult, QubitDensity

from emu_sv.sv_config import SVConfig, StateVector
from emu_sv.sv_backend import SVBackend


import pulser.noise_model

from test.utils_testing import (
    pulser_afm_sequence_ring,
)


seed = 1337

sv_backend = SVBackend()


Omega_max = 4 * 2 * torch.pi
U = Omega_max / 2
delta_0 = -6 * U
delta_f = 2 * U
t_rise = 500
t_fall = 1000


def create_antiferromagnetic_state_vector(num_qubits: int) -> StateVector:
    factors = [torch.zeros(2, dtype=torch.complex128) for _ in range(num_qubits)]
    for i in range(num_qubits):
        if i % 2:
            factors[i][0] = 1.0
        else:
            factors[i][1] = 1.0

    afm_vec = factors[0]
    for i in factors[1:]:
        afm_vec = torch.kron(afm_vec, i)

    return StateVector(afm_vec)


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
    final_time = seq.get_duration()

    if given_fidelity_state:
        fidelity_state = create_antiferromagnetic_state_vector(
            len(seq.register.qubit_ids)
        )
    else:
        fidelity_state = StateVector.make(len(seq.register.qubit_ids))

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

    nqubits = len(seq.register.qubit_ids)
    times = {final_time}

    sv_config = SVConfig(
        initial_state=initial_state,
        dt=dt,
        krylov_tolerance=1e-5,
        observables=[
            StateResult(evaluation_times=times),
            BitStrings(evaluation_times=times, num_shots=1000),
            Fidelity(evaluation_times=times, state=fidelity_state),
            QubitDensity(evaluation_times=times, basis={"r", "g"}, nqubits=nqubits),
        ],
        noise_model=noise_model,
        interaction_cutoff=interaction_cutoff,
    )

    result = sv_backend.run(seq, sv_config)

    return result


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
        f"fidelity_{emu_base.base_classes.default_callbacks._fidelity_counter}"
    ][final_time]

    fidelity_state = create_antiferromagnetic_state_vector(num_qubits)

    assert bitstrings["1010101010"] == 136
    assert bitstrings["0101010101"] == 159
    assert fidelity_state.inner(final_state) == approx(final_fidelity, abs=1e-10)

    q_density = result["qubit_density"][final_time]
    q_density = torch.tensor(q_density, dtype=torch.float64)
    # 0.578 is the magnetization in the first atom
    assert torch.allclose(
        torch.tensor([0.578] * 10, dtype=torch.float64), q_density, atol=1e-3
    )
