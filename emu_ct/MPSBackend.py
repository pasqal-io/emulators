from pulser import Sequence

from emu_ct.mps import MPS
from emu_ct.hamiltonian import make_H, rydberg_interaction
from emu_ct.tdvp import evolve_tdvp
from emu_ct.MPSConfig import MPSConfig
from emu_ct.base_classes.config import BackendConfig
from emu_ct.base_classes.backend import Backend
from emu_ct.pulser_adapter import _extract_omega_delta
from emu_ct.base_classes.results import Results


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
        # TODO: dt is in 2 places: omega_delta and evolve_tdvp,
        coeff = 0.001  # Omega and delta are given in rad/ms, dt in ns
        dt = mps_config.dt
        omega_delta = _extract_omega_delta(sequence, dt, mps_config.with_modulation)

        if mps_config.interaction_matrix is not None:
            assert len(sequence.register.qubit_ids) == mps_config.interaction_matrix.size(
                dim=1
            ), (
                "The number of qubits in the register should be the same as the dimension of "
                "the columns of the interaction matrix"
            )

        interaction_matrix = (
            rydberg_interaction(sequence)
            if mps_config.interaction_matrix is None
            else mps_config.interaction_matrix
        )

        evolve_state = MPS(
            len(sequence.register.qubits),
            truncate=False,
            precision=mps_config.precision,
            max_bond_dim=mps_config.max_bond_dim,
            num_devices_to_use=mps_config.num_devices_to_use,
        )  # TODO: create a config class and take the initial state from it

        result = Results()

        for step in range(omega_delta.shape[1]):
            t = step * dt
            mpo_t = make_H(
                interaction_matrix,
                omega_delta[0, step, :],
                omega_delta[1, step, :],
                mps_config.num_devices_to_use,
            )
            evolve_tdvp(-coeff * dt * 1j, evolve_state, mpo_t, mps_config.max_krylov_dim)

            for callback in mps_config.callbacks:
                callback(t + dt, evolve_state, mpo_t, result)

        return result
