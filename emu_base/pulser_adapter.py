from typing import Sequence, Iterator
from dataclasses import dataclass
from enum import Enum
import math
import torch
import pulser
from pulser.sampler import SequenceSamples
from pulser.noise_model import NoiseModel
from pulser.register.base_register import QubitId
from pulser.backend.config import EmulationConfig
from pulser._hamiltonian_data import HamiltonianData
from pulser.channels.base_channel import States
from emu_base.jump_lindblad_operators import get_lindblad_operators
from emu_base.math.pchip_torch import PCHIP1D


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


def _unique_observable_times(
    config: EmulationConfig,
) -> set[float]:
    """Collect unique evaluation times in [0, 1] for all observables."""
    observable_times: set[float] = set()

    for obs in config.observables:
        if obs.evaluation_times is not None:
            observable_times |= set(obs.evaluation_times)
        elif not isinstance(config.default_evaluation_times, str):  # != "Full"
            observable_times |= set(config.default_evaluation_times.tolist())
        else:
            raise ValueError(
                f"default config {config.default_evaluation_times} is not supported."
            )

    return observable_times


def _get_target_times(
    sequence: pulser.Sequence,
    config: EmulationConfig,
    dt: float,
) -> list[float]:
    """Compute the sorted absolute times to sample the sequence.

    Combines a uniform grid with step ``dt`` and any extra observable times,
    then converts everything to absolute times over the sequence duration.
    """
    duration = float(sequence.get_duration(include_fall_time=config.with_modulation))
    n_steps = math.floor(duration / dt)
    evolution_times_rel: set[float] = {
        i * float(dt) / duration for i in range(n_steps + 1)
    }
    evolution_times_rel.add(1.0)
    target_times_rel = evolution_times_rel | _unique_observable_times(config)
    target_times: list[float] = sorted({t * duration for t in target_times_rel})
    return target_times


def _extract_omega_delta_phi(
    noisy_samples: SequenceSamples,
    qubit_ids: tuple[str, ...],
    target_times: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract per-qubit laser parameters (Ω, δ, phase) from Pulser samples.

    Pulser stores samples on the discrete grid t = 0, 1, ..., T-1
    (with dt = 1.0), i.e. it does not provide values exactly
    at t = T = pulse_duration. Pulser effectively assumes
    Ω(T) = δ(T) = phase(T) = 0. For midpoint discretization we therefore
    interpolate (and implicitly extrapolate near the end) to obtain
    Ω(t_mid), δ(t_mid), and phase(t_mid) at t_mid = (t_k + t_{k+1}) / 2.

    We evaluate the laser parameters at time midpoints to benefit from
    a midpoint scheme.
    https://en.wikipedia.org/wiki/Midpoint_method
    """
    sequence_dict = noisy_samples.to_nested_dict(
        all_local=True,
        samples_type="tensor",
    )["Local"]
    if len(sequence_dict) != 1:
        raise ValueError("Only single interaction type is supported.")

    if "ground-rydberg" in sequence_dict:
        locals_a_d_p = sequence_dict["ground-rydberg"]
    elif "XY" in sequence_dict:
        locals_a_d_p = sequence_dict["XY"]
    else:
        raise ValueError(
            "Only `ground-rydberg` and `mw_global`(XY) channels are supported."
        )
    qubit_ids_filtered = [qid for qid in qubit_ids if qid in locals_a_d_p]

    target_t = torch.as_tensor(target_times, dtype=torch.float64)
    t_mid = 0.5 * (target_t[:-1] + target_t[1:])

    shape = (t_mid.numel(), len(qubit_ids_filtered))
    omega_mid = torch.zeros(shape, dtype=torch.float64, device=t_mid.device)
    delta_mid = torch.zeros(shape, dtype=torch.float64, device=t_mid.device)
    phi_mid = torch.zeros(shape, dtype=torch.float64, device=t_mid.device)

    assert noisy_samples.max_duration == target_times[-1]
    t_grid = torch.arange(target_times[-1], dtype=torch.float64)

    laser_by_data = {
        "amp": omega_mid,
        "det": delta_mid,
        "phase": phi_mid,
    }
    for name, data_mid in laser_by_data.items():
        for q_pos, q_id in enumerate(qubit_ids_filtered):
            signal = torch.as_tensor(locals_a_d_p[q_id][name])
            if torch.is_complex(signal) and not torch.allclose(
                signal.imag, torch.zeros_like(signal.imag)
            ):
                raise ValueError(f"Input {name} has non-zero imaginary part.")

            pchip = PCHIP1D(t_grid, signal.real)
            data_mid[:, q_pos] = pchip(t_mid)
            if name == "amp":
                data_mid[-1, q_pos] = torch.where(
                    data_mid[-1, q_pos] > 0,
                    data_mid[-1, q_pos],
                    0,
                )

    omega_c, delta_c, phi_c = (
        arr.to(torch.complex128) for arr in (omega_mid, delta_mid, phi_mid)
    )
    return omega_c, delta_c, phi_c


@dataclass(frozen=True)
class SequenceData:
    omega: torch.Tensor
    delta: torch.Tensor
    phi: torch.Tensor
    full_interaction_matrix: torch.Tensor
    masked_interaction_matrix: torch.Tensor
    bad_atoms: dict[str, bool]
    lindblad_ops: list[torch.Tensor]
    noise_model: pulser.NoiseModel
    qubit_ids: tuple[QubitId, ...]
    target_times: list[float]
    eigenstates: list[States]
    qubit_count: int
    dim: int
    hamiltonian_type: HamiltonianType
    slm_end_time: float


class PulserData:
    target_times: list[float]
    slm_end_time: float
    full_interaction_matrix: torch.Tensor | None
    hamiltonian_type: HamiltonianType
    lindblad_ops: list[torch.Tensor]
    qubit_ids: tuple[QubitId, ...]
    noise_model: pulser.NoiseModel
    interaction_cutoff: float
    eigenstates: list[States]
    qubit_count: int
    dim: int

    def __init__(self, *, sequence: pulser.Sequence, config: EmulationConfig, dt: float):
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

        self.full_interaction_matrix = None
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
