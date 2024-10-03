import random
from resource import RUSAGE_SELF, getrusage
from time import time
from typing import Optional

import torch
from pulser import Sequence

from emu_mps.base_classes.backend import Backend
from emu_mps.base_classes.config import BackendConfig
from emu_mps.base_classes.results import Results
from emu_mps.hamiltonian import make_H, rydberg_interaction
from emu_mps.math.brents_root_finding import find_root_brents
from emu_mps.mpo import MPO
from emu_mps.mps import MPS
from emu_mps.mps_config import MPSConfig
from emu_mps.noise import compute_noise_from_lindbladians, pick_well_prepared_qubits
from emu_mps.pulser_adapter import (
    extract_omega_delta_phi,
    get_all_lindblad_noise_operators,
)
from emu_mps.tdvp import evolve_tdvp
from emu_mps.utils import (
    extended_mpo_factors,
    extended_mps_factors,
    get_extended_site_index,
)


class _RunImpl:
    well_prepared_qubits_filter: Optional[list[bool]]
    lindblad_ops: list[torch.Tensor]
    lindblad_noise: torch.Tensor
    hamiltonian: MPO
    collapse_threshold: float
    aggregated_lindblad_ops: Optional[torch.Tensor] = None
    is_noisy: bool

    def __init__(self, sequence: Sequence, mps_config: MPSConfig):
        self.sequence = sequence
        self.config = mps_config

        self.dt = self.config.dt
        self.current_time: float = (
            0.0  # While dt is an integer, noisy collapse can happen at non-integer times.
        )
        self.qubit_count = len(sequence.register.qubit_ids)

        assert (
            mps_config.interaction_matrix is None
            or mps_config.interaction_matrix.size(dim=0) == self.qubit_count
        ), (
            "The number of qubits in the register should be the same as the size of "
            "the interaction matrix"
        )

        self.omega, self.delta, self.phi = extract_omega_delta_phi(
            self.sequence, self.dt, self.config.with_modulation
        )
        self.timestep_count = self.omega.shape[0]

        if self.config.interaction_matrix is None:
            self.interaction_matrix = rydberg_interaction(sequence)

        else:
            self.interaction_matrix = self.config.interaction_matrix

        self.init_dark_qubits()
        self.init_lindblad_noise()

        self.state = self.make_initial_state()

    def init_dark_qubits(self) -> None:
        has_state_preparation_error: bool = (
            self.config.noise_model is not None
            and "SPAM" in self.config.noise_model.noise_types
            and self.config.noise_model.state_prep_error != 0.0
        )

        self.well_prepared_qubits_filter = (
            pick_well_prepared_qubits(
                self.config.noise_model.state_prep_error, self.qubit_count
            )
            if has_state_preparation_error
            else None
        )

        if self.well_prepared_qubits_filter is not None:
            self.interaction_matrix = self.interaction_matrix[
                self.well_prepared_qubits_filter, :
            ][:, self.well_prepared_qubits_filter]
            self.omega = self.omega[:, self.well_prepared_qubits_filter]
            self.delta = self.delta[:, self.well_prepared_qubits_filter]
            self.phi = self.phi[:, self.well_prepared_qubits_filter]

    def init_lindblad_noise(self) -> None:
        self.lindblad_ops = get_all_lindblad_noise_operators(self.config.noise_model)

        self.is_noisy = self.lindblad_ops != []

        if self.is_noisy:
            stacked = torch.stack(self.lindblad_ops)
            # The below is used for batch computation of noise collapse weights.
            self.aggregated_lindblad_ops = stacked.conj().transpose(1, 2) @ stacked

        self.lindblad_noise = compute_noise_from_lindbladians(self.lindblad_ops)
        self.collapse_threshold = random.random()

    def make_initial_state(self) -> MPS:
        if self.config.initial_state is None:
            well_prepared_qubits_count: int = (
                self.qubit_count
                if self.well_prepared_qubits_filter is None
                else sum(1 for x in self.well_prepared_qubits_filter if x)
            )

            return MPS.make(
                well_prepared_qubits_count,
                precision=self.config.precision,
                max_bond_dim=self.config.max_bond_dim,
                num_gpus_to_use=self.config.num_gpus_to_use,
            )

        if self.well_prepared_qubits_filter is not None:
            raise NotImplementedError(
                "Specifying the initial state in the presence of state \
                    preparation errors is currently not implemented."
            )

        assert isinstance(self.config.initial_state, MPS)
        initial_state = MPS(
            # Deep copy of every tensor of the initial state.
            [f.clone().detach() for f in self.config.initial_state.factors],
            precision=self.config.precision,
            max_bond_dim=self.config.max_bond_dim,
            num_gpus_to_use=self.config.num_gpus_to_use,
        )
        initial_state.truncate()
        return initial_state

    def do_time_step(self, step: int) -> None:
        """
        Updates hamiltonian and evolves state by dt.
        """
        self.hamiltonian = make_H(
            interaction_matrix=self.interaction_matrix,
            omega=self.omega[step, :],
            delta=self.delta[step, :],
            phi=self.phi[step, :],
            noise=self.lindblad_noise,
        )

        _TIME_CONVERSION_COEFF = 0.001  # Omega and delta are given in rad/ms, dt in ns

        target_time = float((step + 1) * self.dt)
        while self.current_time != target_time:
            assert self.current_time < target_time

            time_at_begin = self.current_time
            squared_norm_at_begin = self.state.norm() ** 2
            assert self.collapse_threshold <= squared_norm_at_begin

            def evolve_to_time(intermediate_time: float) -> float:
                delta = intermediate_time - self.current_time
                evolve_tdvp(
                    t=-_TIME_CONVERSION_COEFF * delta * 1j,
                    state=self.state,
                    hamiltonian=self.hamiltonian,
                    extra_krylov_tolerance=self.config.extra_krylov_tolerance,
                    max_krylov_dim=self.config.max_krylov_dim,
                    is_hermitian=not self.is_noisy,
                )
                self.current_time = intermediate_time
                return self.state.norm() ** 2 - self.collapse_threshold

            delta_collapse = evolve_to_time(target_time)

            if self.is_noisy and delta_collapse < 0:
                find_root_brents(
                    f=evolve_to_time,
                    start=time_at_begin,
                    end=target_time,
                    f_start=squared_norm_at_begin - self.collapse_threshold,
                    f_end=delta_collapse,
                    tolerance=1.0,
                )
                self.random_noise_collapse()

        assert self.current_time == target_time

    def random_noise_collapse(self) -> None:
        collapse_weights = self.state.expect_batch(self.aggregated_lindblad_ops).real

        ((collapsed_qubit_index, collapse_operator),) = random.choices(
            [
                (qubit, op)
                for qubit in range(self.state.num_sites)
                for op in self.lindblad_ops
            ],
            collapse_weights.reshape(-1),
        )

        self.state.apply(collapsed_qubit_index, collapse_operator)
        self.state = (1 / self.state.norm()) * self.state

        assert abs(1 - self.state.norm()) < 1e-10

        self.collapse_threshold = random.random()

    def fill_results(self, results: Results, step: int) -> None:
        t = (step + 1) * self.dt  # we are now after the time-step, so use step+1

        noiseless_hamiltonian = (
            self.hamiltonian
            if not self.is_noisy
            else make_H(
                interaction_matrix=self.interaction_matrix,
                omega=self.omega[step, :],
                delta=self.delta[step, :],
                phi=self.phi[step, :],
                # Without noise for the callbacks.
            )
        )

        normalized_state = 1 / self.state.norm() * self.state

        if self.well_prepared_qubits_filter is None:
            for callback in self.config.callbacks:
                callback(self.config, t, normalized_state, noiseless_hamiltonian, results)
            return

        full_mpo, full_state = None, None
        for callback in self.config.callbacks:
            if t not in callback.evaluation_times:
                continue

            if full_mpo is None or full_state is None:
                # Only do this potentially expensive step once and when needed.
                full_mpo = MPO(
                    extended_mpo_factors(
                        noiseless_hamiltonian.factors, self.well_prepared_qubits_filter
                    )
                )
                full_state = MPS(
                    extended_mps_factors(
                        normalized_state.factors, self.well_prepared_qubits_filter
                    ),
                    num_gpus_to_use=None,  # Keep the already assigned devices.
                    orthogonality_center=get_extended_site_index(
                        self.well_prepared_qubits_filter,
                        normalized_state.orthogonality_center,
                    ),
                )

            callback(self.config, t, full_state, full_mpo, results)

    def log_step_statistics(self, *, step: int, duration: float) -> None:
        if self.state.factors[0].is_cuda:
            max_mem_per_device = (
                torch.cuda.max_memory_allocated(device) * 1e-6
                for device in range(torch.cuda.device_count())
            )
            max_mem = max(max_mem_per_device)
        else:
            max_mem = getrusage(RUSAGE_SELF).ru_maxrss * 1e-3

        self.config.logger.info(
            f"step = {step + 1}/{self.timestep_count}, "
            + f"χ = {self.state.get_max_bond_dim()}, "
            + f"|ψ| = {self.state.get_memory_footprint():.3f} MB, "
            + f"RSS = {max_mem:.3f} MB, "
            + f"Δt = {duration:.3f} s"
        )


class MPSBackend(Backend):
    """
    A backend for emulating Pulser sequences using Matrix Product States (MPS),
    aka tensor trains.
    """

    def run(self, sequence: Sequence, mps_config: BackendConfig) -> Results:
        """
        Emulates the given sequence.

        Args:
            sequence: a Pulser sequence to simulate
            mps_config: the backends config. Should be of type MPSConfig

        Returns:
            the simulation results
        """
        assert isinstance(mps_config, MPSConfig)

        self.validate_sequence(sequence)

        results = Results()

        impl = _RunImpl(sequence, mps_config)

        for step in range(impl.timestep_count):
            start = time()

            impl.do_time_step(step)

            impl.fill_results(results, step)

            end = time()
            impl.log_step_statistics(step=step, duration=end - start)

        return results
