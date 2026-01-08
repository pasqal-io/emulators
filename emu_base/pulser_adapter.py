from typing import Sequence, Iterator
from dataclasses import dataclass
from enum import Enum
import torch
import math
import pulser
from pulser.sampler import SequenceSamples
from pulser.noise_model import NoiseModel
from pulser.register.base_register import QubitId
from pulser.backend.config import EmulationConfig
from pulser._hamiltonian_data import HamiltonianData
from emu_base.jump_lindblad_operators import get_lindblad_operators


class HamiltonianType(Enum):
    Rydberg = 1
    XY = 2


_NON_LINDBLADIAN_NOISE = {"SPAM", "doppler", "amplitude", "detuning", "register"}


def _get_all_lindblad_noise_operators(
    noise_model: NoiseModel | None, dim: int = 2, interact_type: str = "ising"
) -> list[torch.Tensor]:
    if noise_model is None:
        return []

    return [
        op
        for noise_type in noise_model.noise_types
        if noise_type not in _NON_LINDBLADIAN_NOISE
        for op in get_lindblad_operators(
            noise_type=noise_type,
            noise_model=noise_model,
            dim=dim,
            interact_type=interact_type,
        )
    ]


def _get_target_times(
    sequence: pulser.Sequence, config: EmulationConfig, dt: int
) -> list[int]:
    sequence_duration = sequence.get_duration(include_fall_time=config.with_modulation)

    observable_times = set(range(0, sequence_duration, dt))
    observable_times.add(sequence_duration)
    for obs in config.observables:
        times: Sequence[float]
        if obs.evaluation_times is not None:
            times = obs.evaluation_times
        elif config.default_evaluation_times != "Full":
            times = config.default_evaluation_times.tolist()  # type: ignore[union-attr,assignment]
        observable_times |= set([round(time * sequence_duration) for time in times])

    target_times: list[int] = list(observable_times)
    target_times.sort()
    return target_times


def _extract_omega_delta_phi(
    noisy_samples: SequenceSamples,
    qubit_ids: tuple[str, ...],
    target_times: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_dict = noisy_samples.to_nested_dict(all_local=True, samples_type="tensor")[
        "Local"
    ]
    nsamples = len(target_times) - 1
    omega = torch.zeros(
        nsamples,
        len(qubit_ids),
        dtype=torch.complex128,
    )
    delta = torch.zeros(
        nsamples,
        len(qubit_ids),
        dtype=torch.complex128,
    )
    phi = torch.zeros(
        nsamples,
        len(qubit_ids),
        dtype=torch.complex128,
    )
    max_duration = noisy_samples.max_duration

    if "ground-rydberg" in sequence_dict and len(sequence_dict) == 1:
        locals_a_d_p = sequence_dict["ground-rydberg"]
    elif "XY" in sequence_dict and len(sequence_dict) == 1:
        locals_a_d_p = sequence_dict["XY"]
    else:
        raise ValueError("Only `ground-rydberg` and `mw_global` channels are supported.")
    qubit_ids_filtered = [qid for qid in qubit_ids if qid in locals_a_d_p]
    for i in range(nsamples):
        t = (target_times[i] + target_times[i + 1]) / 2
        # The sampled values correspond to the start of each interval
        # To maximize the order of the solver, we need the values in the middle
        if math.ceil(t) < max_duration:
            # If we're not the final step, approximate this using linear
            # interpolation
            # Note that for dt even, t1=t2
            t1 = math.floor(t)
            t2 = math.ceil(t)
            for q_pos, q_id in enumerate(qubit_ids_filtered):
                omega[i, q_pos] = (
                    locals_a_d_p[q_id]["amp"][t1] + locals_a_d_p[q_id]["amp"][t2]
                ) / 2.0
                delta[i, q_pos] = (
                    locals_a_d_p[q_id]["det"][t1] + locals_a_d_p[q_id]["det"][t2]
                ) / 2.0
                phi[i, q_pos] = (
                    locals_a_d_p[q_id]["phase"][t1] + locals_a_d_p[q_id]["phase"][t2]
                ) / 2.0

        else:
            # We're in the final step and dt=1, approximate this using linear extrapolation
            # we can reuse omega_1 and omega_2 from before
            for q_pos, q_id in enumerate(qubit_ids_filtered):
                delta[i, q_pos] = (
                    3.0 * locals_a_d_p[q_id]["det"][t2] - locals_a_d_p[q_id]["det"][t1]
                ) / 2.0
                phi[i, q_pos] = (
                    3.0 * locals_a_d_p[q_id]["phase"][t2]
                    - locals_a_d_p[q_id]["phase"][t1]
                ) / 2.0
                omega[i, q_pos] = max(
                    (3.0 * locals_a_d_p[q_id]["amp"][t2] - locals_a_d_p[q_id]["amp"][t1])
                    / 2.0,
                    0.0,
                )

    return omega, delta, phi


@dataclass(frozen=True)
class SequenceData:
    omega: torch.Tensor
    delta: torch.Tensor
    phi: torch.Tensor
    full_interaction_matrix: torch.Tensor
    masked_interaction_matrix: torch.Tensor
    bad_atoms: list
    lindblad_ops: list[torch.Tensor]
    noise_model: pulser.NoiseModel
    qubit_ids: tuple[QubitId, ...]
    target_times: list[int]
    eigenstates: list[str]
    qubit_count: int
    dim: int
    hamiltonian_type: HamiltonianType
    slm_end_time: float


class PulserData:
    target_times: list[int]
    slm_end_time: float
    full_interaction_matrix: torch.Tensor
    masked_interaction_matrix: torch.Tensor
    hamiltonian_type: HamiltonianType
    lindblad_ops: list[torch.Tensor]
    qubit_ids: tuple[QubitId, ...]
    noise_model: pulser.NoiseModel
    interaction_cutoff: float
    eigenstates: list[str]
    qubit_count: int
    dim: int

    def __init__(self, *, sequence: pulser.Sequence, config: EmulationConfig, dt: int):
        self._sequence = sequence
        self.qubit_ids = sequence.register.qubit_ids
        self.qubit_count = len(self.qubit_ids)
        self.target_times = _get_target_times(sequence=sequence, config=config, dt=dt)
        self.noise_model = (
            sequence.device.default_noise_model
            if config.prefer_device_noise_model
            else config.noise_model
        )

        if not self.noise_model:
            self.noise_model = NoiseModel()

        self.hamiltonian = HamiltonianData.from_sequence(
            sequence,
            with_modulation=config.with_modulation,
            noise_model=self.noise_model,
            n_trajectories=config.n_trajectories,
        )

        self.eigenstates = self.hamiltonian.basis_data.eigenbasis

        int_type = self.hamiltonian.basis_data.interaction_type
        self.dim = self.hamiltonian.basis_data.dim
        if int_type == "ising":  # for local and global
            self.hamiltonian_type = HamiltonianType.Rydberg
        elif int_type == "XY":
            self.hamiltonian_type = HamiltonianType.XY
        else:
            raise ValueError(f"Unsupported basis: {int_type}")

        self.lindblad_ops = _get_all_lindblad_noise_operators(
            self.noise_model, dim=self.dim, interact_type=int_type
        )
        self.has_lindblad_noise: bool = self.lindblad_ops != []

        self.full_interaction_matrix: torch.Tensor | None = None
        if config.interaction_matrix is not None:
            assert len(config.interaction_matrix) == self.qubit_count, (
                "The number of qubits in the register should be the same as the size of "
                "the interaction matrix"
            )
            self.full_interaction_matrix = config.interaction_matrix.as_tensor()

        self.interaction_cutoff = config.interaction_cutoff
        self.slm_end_time = (
            sequence._slm_mask_time[1] if len(sequence._slm_mask_time) > 1 else 0.0
        )

    def get_sequences(self) -> Iterator[SequenceData]:
        for samples in self.hamiltonian.noisy_samples:
            full_interaction_matrix = (
                self.full_interaction_matrix
                if self.full_interaction_matrix is not None
                else samples.trajectory.interaction_matrix.as_tensor()
            )

            full_interaction_matrix[
                torch.abs(full_interaction_matrix) < self.interaction_cutoff
            ] = 0.0

            masked_interaction_matrix = full_interaction_matrix.clone()

            # disable interaction for SLM masked qubits

            slm_targets = list(self._sequence._slm_mask_targets)
            for target in self._sequence.register.find_indices(slm_targets):
                masked_interaction_matrix[target] = 0.0
                masked_interaction_matrix[:, target] = 0.0

            omega, delta, phi = _extract_omega_delta_phi(
                samples.samples, self.qubit_ids, self.target_times
            )

            for _ in range(samples.reps):
                yield SequenceData(
                    omega,
                    delta,
                    phi,
                    full_interaction_matrix,
                    masked_interaction_matrix,
                    samples.trajectory.bad_atoms,
                    self.lindblad_ops,
                    self.noise_model,
                    self.qubit_ids,
                    self.target_times,
                    self.eigenstates,
                    self.qubit_count,
                    self.dim,
                    self.hamiltonian_type,
                    self.slm_end_time,
                )
