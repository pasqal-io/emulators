from pulser import Sequence

from emu_ct.mps import MPS
from emu_ct.hamiltonian import make_H
from emu_ct.tdvp import evolve_tdvp
from .MPSConfig import MPSConfig
from .base_classes.config import BackendConfig


from .base_classes.backend import Backend
from .pulser_adapter import get_qubit_positions, _extract_omega_delta

from .base_classes.results import Results


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
        emuct_register = get_qubit_positions(sequence.register)

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
                emuct_register,
                omega_delta[0, step, :],
                omega_delta[1, step, :],
                sequence.device.interaction_coeff,
            )
            evolve_tdvp(-coeff * dt * 1j, evolve_state, mpo_t, mps_config.max_krylov_dim)
            for callback in mps_config.callbacks:
                callback(t + dt, evolve_state, mpo_t, result)

        return result
