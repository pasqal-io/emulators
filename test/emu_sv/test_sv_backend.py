from unittest.mock import MagicMock
import torch
from emu_sv import SVConfig, SVBackend, Occupation
from emu_sv.sv_backend_impl import SVBackendImpl
from emu_base import SequenceData, HamiltonianType
import pytest
import logging

device = "cpu"


def test_sv_impl():
    """test that index_add is called in a no_grad context in forward"""
    config = SVConfig(gpu=False if device == "cpu" else True)
    pulser_data = MagicMock(
        spec=SequenceData,
        omega=torch.tensor([[1.0]], requires_grad=True),
        delta=torch.tensor([[1.0]]),
        phi=torch.tensor([[1.0]]),
        interaction_matrix=lambda t: torch.tensor(0.0),
        state_prep_error=0.0,
        target_times=[1.0],
        qubit_ids=(),
        lindblad_ops=[],
    )
    bknd_impl = SVBackendImpl(config, pulser_data)
    bknd_impl._evolve_step(1.0, 0)


def test_run_from_sequence_data():
    duration = 100
    dt = 10

    occup = Occupation(
        evaluation_times=[dt * x / duration for x in range(duration // dt + 1)]
    )

    config = SVConfig(
        observables=[occup],
        log_level=logging.WARN,
        interaction_cutoff=1e-10,
    )

    omega = torch.tensor(
        [
            [0.7092 + 0.0j, 0.7092 + 0.0j, 0.7092 + 0.0j],
            [7.8431 + 0.0j, 7.8431 + 0.0j, 7.8431 + 0.0j],
            [26.2913 + 0.0j, 26.2913 + 0.0j, 26.2913 + 0.0j],
            [53.0011 + 0.0j, 53.0011 + 0.0j, 53.0011 + 0.0j],
            [73.0656 + 0.0j, 73.0656 + 0.0j, 73.0656 + 0.0j],
            [71.8630 + 0.0j, 71.8630 + 0.0j, 71.8630 + 0.0j],
            [50.3238 + 0.0j, 50.3238 + 0.0j, 50.3238 + 0.0j],
            [23.9187 + 0.0j, 23.9187 + 0.0j, 23.9187 + 0.0j],
            [6.6745 + 0.0j, 6.6745 + 0.0j, 6.6745 + 0.0j],
            [0.4483 + 0.0j, 0.4483 + 0.0j, 0.4483 + 0.0j],
        ],
        dtype=torch.complex128,
    )

    seq_data = SequenceData(
        omega=omega,
        delta=torch.zeros_like(omega),
        phi=torch.zeros_like(omega),
        interaction_matrix=lambda x: torch.zeros((3, 3), dtype=torch.float64),
        qubit_ids=("q0", "q1", "a2"),
        bad_atoms=(False, False, False),
        lindblad_ops=[],
        state_prep_error=0.0,
        target_times=[dt * x for x in range(duration // dt + 1)],
        eigenstates=["r", "g"],
        hamiltonian_type=HamiltonianType.Rydberg,
    )

    results = SVBackend._run_from_sequence_data(seq_data, config)

    assert set(results._tagmap.keys()) == {"occupation", "statistics"}
    assert results.get_result_times("occupation") == occup.evaluation_times.tolist()
    assert results.occupation[0] == pytest.approx(0.0)
    assert results.occupation[-1] == pytest.approx(1.0)
