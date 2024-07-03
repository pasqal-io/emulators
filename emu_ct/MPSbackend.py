import warnings

import torch


from pulser.backend.abc import Backend
import pulser


from emu_ct.mps import MPS
from emu_ct.tdvp import evolve_tdvp
from emu_ct.hamiltonian import make_H
from emu_ct.results import MPSBackendResults
from emu_ct.pulser_adapter import get_qubit_positions, _extract_omega_delta


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

    def __init__(
        self,
        sequence: pulser.Sequence,
    ):
        """Initializes a new MPSBackend."""
        super().__init__(sequence)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)

        if not isinstance(sequence, pulser.Sequence):
            raise TypeError(
                "The provided sequence has to be a valid " "pulser.Sequence instance."
            )
        if sequence.is_parametrized() or sequence.is_register_mappable():
            raise ValueError(
                "Not supported"
                "The provided sequence needs to be built to be simulated. Call"
                " `Sequence.build()` with the necessary parameters."
            )
        if not sequence._schedule:
            raise ValueError("The provided sequence has no declared channels.")
        if all(sequence._schedule[x][-1].tf == 0 for x in sequence.declared_channels):
            raise ValueError("No instructions given for the channels in the sequence.")

        self.sequence = sequence

    def run(self, dt: int = 1, with_modulation: bool = False) -> MPSBackendResults:
        """Emulates the sequences using Emu_ct solvers.

        Returns:
            MPSBackendResults
        """

        # dt & with_modulation TODO: create a config class and take those from it
        coeff = 0.001  # Omega and delta are given in rad/ms, dt in ns
        omega_delta = _extract_omega_delta(self.sequence, dt, with_modulation)
        emuct_register = get_qubit_positions(self.sequence.register)

        _evolve_state = MPS(
            [(torch.tensor([1.0, 0.0]).reshape(1, 2, 1).to(dtype=torch.complex128))]
            * len(self.sequence.register.qubits)
        )  # TODO: create a config class and take the initial state from it

        for step in range(omega_delta.shape[1]):
            mpo_t0 = make_H(
                emuct_register, omega_delta[0, step, :], omega_delta[1, step, :]
            )
            evolve_tdvp(-coeff * dt * 1j, _evolve_state, mpo_t0)

        return MPSBackendResults(_evolve_state)
