import torch
from emu_ct import MPS, MPSBackend, MPSConfig, BitStrings, StateResult

from .utils_testing import pulser_afm_sequence_ring, pulser_afm_sequence_grid

from pytest import approx
from unittest.mock import patch
import pytest
import pulser

seed = 1337

mps_backend = MPSBackend()


def simulate(seq, state_prep_error=None):
    final_time = seq.get_duration()
    if state_prep_error is None:
        noise_model = None
    else:
        noise_model = pulser.noise_model.NoiseModel(
            noise_types=("SPAM",),
            state_prep_error=state_prep_error,
            p_false_pos=0.0,
            p_false_neg=0.0,
        )

    mps_config = MPSConfig(
        dt=100,
        precision=1e-5,
        observables=[
            StateResult(times=[final_time]),
            BitStrings(times=[final_time], num_shots=1000),
        ],
        noise_model=noise_model,
    )

    result = mps_backend.run(seq, mps_config)

    final_time = seq.get_duration()
    bitstrings = result[BitStrings.name()][final_time]
    final_state = result[StateResult.name()][final_time]
    return bitstrings, final_state


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

    bitstrings, final_state = simulate(seq)

    assert bitstrings["1010101010"] == 148
    assert bitstrings["0101010101"] == 151

    max_bond_dim = final_state.get_max_bond_dim()
    assert max_bond_dim == 29

    assert get_proba(final_state, "1010101010") == approx(0.15, abs=1e-2)
    assert get_proba(final_state, "0101010101") == approx(0.15, abs=1e-2)


def test_end_to_end_afm_line_with_state_preparation_error():
    torch.manual_seed(seed)

    def simulate_line(n, state_prep_error=0.1):
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
        return simulate(seq, state_prep_error=state_prep_error)

    with patch(
        "emu_ct.mps_backend.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        pick_well_prepared_qubits_mock.return_value = [True, True, True, False]
        bitstrings, final_state = simulate_line(4, state_prep_error=0.1)
        pick_well_prepared_qubits_mock.assert_called_with(0.1, 4)

    assert get_proba(final_state, "1110") == approx(0.51, abs=1e-2)
    assert get_proba(final_state, "1010") == approx(0.46, abs=1e-2)

    # A dark qubit at the end of the line gives the same result as a line with one less qubit.
    with patch(
        "emu_ct.mps_backend.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        bitstrings, final_state = simulate_line(3, state_prep_error=None)
        pick_well_prepared_qubits_mock.assert_not_called()
        assert get_proba(final_state, "111") == approx(0.51, abs=1e-2)
        assert get_proba(final_state, "101") == approx(0.46, abs=1e-2)

    with patch(
        "emu_ct.mps_backend.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        pick_well_prepared_qubits_mock.return_value = [True, False, True, True]
        bitstrings, final_state = simulate_line(4)

    assert get_proba(final_state, "1011") == approx(0.99, abs=1e-2)

    # Results for a 2 qubit line.
    bitstrings, final_state = simulate_line(2, state_prep_error=None)
    assert get_proba(final_state, "11") == approx(0.99, abs=1e-2)

    with patch(
        "emu_ct.mps_backend.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        pick_well_prepared_qubits_mock.return_value = [False, True, True, False]
        bitstrings, final_state = simulate_line(4)

    assert get_proba(final_state, "0110") == approx(0.99, abs=1e-2)

    # FIXME: When n-1 qubits are dark, the simulation fails!
    with patch(
        "emu_ct.mps_backend.pick_well_prepared_qubits"
    ) as pick_well_prepared_qubits_mock:
        with pytest.raises(ValueError) as exception_info:
            pick_well_prepared_qubits_mock.return_value = [False, False, True, False]
            bitstrings, final_state = simulate_line(4)

    assert "For 1 qubit states, do state vector" in str(exception_info.value)
