from pulser import Sequence
from time import time
import torch
from resource import getrusage, RUSAGE_SELF

from emu_ct.mps import MPS
from emu_ct.hamiltonian import make_H, rydberg_interaction
from emu_ct.mpo import MPO
from emu_ct.tdvp import evolve_tdvp
from emu_ct.mps_config import MPSConfig
from emu_ct.base_classes.config import BackendConfig
from emu_ct.base_classes.backend import Backend
from emu_ct.pulser_adapter import _extract_omega_delta
from emu_ct.base_classes.results import Results
from emu_ct.utils import extended_mps_factors, extended_mpo_factors
from emu_ct.noise import pick_well_prepared_qubits


class MPSBackend(Backend):
    """A backend for emulating the sequences using Matrix Product State (MPS).

    Args:
        sequence: An instance of a Pulser Sequence that we
            want to simulate.
        sampling_rate: The fraction of samples that we wish to
            extract from the pulse sequence to simulate. Has to be a
            value between 0.05 and 1.0
        with_modulation: Whether to simulate the sequence with the
            programmed input or the expected output.
    """

    def run(self, sequence: Sequence, mps_config: BackendConfig) -> Results:
        """Emulates the sequences using Emu_ct solvers.

        Returns:
            MPSBackendResults
        """
        assert isinstance(mps_config, MPSConfig)

        self.validate_sequence(sequence)

        coeff = 0.001  # Omega and delta are given in rad/ms, dt in ns
        dt = mps_config.dt
        omega, delta = _extract_omega_delta(sequence, dt, mps_config.with_modulation)

        qubit_count = len(sequence.register.qubit_ids)

        if mps_config.interaction_matrix is not None:
            assert mps_config.interaction_matrix.size(dim=1) == qubit_count, (
                "The number of qubits in the register should be the same as the dimension of "
                "the columns of the interaction matrix"
            )

        interaction_matrix = (
            rydberg_interaction(sequence)
            if mps_config.interaction_matrix is None
            else mps_config.interaction_matrix
        )

        has_dark_qubit: bool = (
            mps_config.noise_model is not None
            and "SPAM" in mps_config.noise_model.noise_types
            and mps_config.noise_model.state_prep_error != 0.0
        )

        well_prepared_qubits_filter = (
            pick_well_prepared_qubits(
                mps_config.noise_model.state_prep_error, len(sequence.register.qubits)
            )
            if has_dark_qubit
            else slice(None, None, None)
        )

        well_prepared_qubits_count: int = (
            sum(1 for x in well_prepared_qubits_filter if x)  # type: ignore
            if has_dark_qubit
            else qubit_count
        )

        state = MPS(
            well_prepared_qubits_count,
            truncate=False,
            precision=mps_config.precision,
            max_bond_dim=mps_config.max_bond_dim,
            num_devices_to_use=mps_config.num_devices_to_use,
        )  # TODO: take the initial state from configuration.

        result = Results()

        for step in range(omega.shape[0]):
            start = time()
            mpo = make_H(
                interaction_matrix[well_prepared_qubits_filter, :][
                    :, well_prepared_qubits_filter
                ],
                omega[step, well_prepared_qubits_filter],
                delta[step, well_prepared_qubits_filter],
                mps_config.num_devices_to_use,
            )
            evolve_tdvp(-coeff * dt * 1j, state, mpo, mps_config.max_krylov_dim)

            t = (step + 1) * dt  # we are now after the time-step, so use step+1
            for callback in mps_config.callbacks:
                if not has_dark_qubit:
                    callback(t, state, mpo, result)
                elif t in callback.times:
                    assert isinstance(well_prepared_qubits_filter, list)  # For mypy.
                    full_mpo = MPO(
                        extended_mpo_factors(mpo.factors, well_prepared_qubits_filter)
                    )
                    full_state = MPS(
                        extended_mps_factors(state.factors, well_prepared_qubits_filter),
                        keep_devices=True,
                    )
                    callback(t, full_state, full_mpo, result)
            end = time()
            mem = (
                torch.cuda.max_memory_allocated() * 1e-6
                if state.factors[0].is_cuda
                else getrusage(RUSAGE_SELF).ru_maxrss * 1e-3
            )
            print(
                f"step = {step}/{omega.shape[0]},",
                f"χ = {state.get_max_bond_dim()},",
                f"|ψ| = {state.get_memory_footprint():.3f} MB,",
                f"RSS = {mem:.3f} MB,",
                f"Δt = {end - start} s",
            )

        return result
