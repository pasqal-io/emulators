import math
import random
from resource import RUSAGE_SELF, getrusage
from typing import Optional

import torch
from pulser import Sequence

from emu_base import Results, State, find_root_brents, PulserData
from emu_mps.hamiltonian import make_H
from emu_mps.mpo import MPO
from emu_mps.mps import MPS
from emu_mps.mps_config import MPSConfig
from emu_mps.noise import compute_noise_from_lindbladians, pick_well_prepared_qubits
from emu_mps.tdvp import evolve_tdvp
from emu_mps.utils import (
    extended_mpo_factors,
    extended_mps_factors,
    get_extended_site_index,
)


class MPSBackendImpl:
    current_time: float = (
        0.0  # While dt is an integer, noisy collapse can happen at non-integer times.
    )
    well_prepared_qubits_filter: Optional[list[bool]]
    lindblad_ops: list[torch.Tensor]
    lindblad_noise: torch.Tensor
    hamiltonian: MPO
    jump_threshold: float
    aggregated_lindblad_ops: Optional[torch.Tensor] = None
    has_lindblad_noise: bool
    norm_gap_before_jump: float
    state: MPS

    def __init__(self, sequence: Sequence, mps_config: MPSConfig):
        self.config = mps_config
        self.qubit_count = len(sequence.register.qubit_ids)

        assert (
            mps_config.interaction_matrix is None
            or len(mps_config.interaction_matrix) == self.qubit_count
        ), (
            "The number of qubits in the register should be the same as the size of "
            "the interaction matrix"
        )

        pulser_data = PulserData(sequence=sequence, config=mps_config, dt=mps_config.dt)
        self.omega = pulser_data.omega
        self.delta = pulser_data.delta
        self.phi = pulser_data.phi
        self.timestep_count = self.omega.shape[0]
        self.full_interaction_matrix = pulser_data.full_interaction_matrix
        self.masked_interaction_matrix = pulser_data.masked_interaction_matrix
        self.hamiltonian_type = pulser_data.hamiltonian_type
        self.slm_end_time = pulser_data.slm_end_time
        self.lindblad_ops = pulser_data.lindblad_ops

    def init_dark_qubits(self) -> None:
        has_state_preparation_error: bool = (
            self.config.noise_model is not None
            and self.config.noise_model.state_prep_error > 0.0
        )

        self.well_prepared_qubits_filter = (
            pick_well_prepared_qubits(
                self.config.noise_model.state_prep_error, self.qubit_count
            )
            if has_state_preparation_error
            else None
        )

        if self.well_prepared_qubits_filter is not None:
            self.full_interaction_matrix = self.full_interaction_matrix[
                self.well_prepared_qubits_filter, :
            ][:, self.well_prepared_qubits_filter]
            self.masked_interaction_matrix = self.masked_interaction_matrix[
                self.well_prepared_qubits_filter, :
            ][:, self.well_prepared_qubits_filter]
            self.omega = self.omega[:, self.well_prepared_qubits_filter]
            self.delta = self.delta[:, self.well_prepared_qubits_filter]
            self.phi = self.phi[:, self.well_prepared_qubits_filter]

    def init_lindblad_noise(self) -> None:
        self.has_lindblad_noise = self.lindblad_ops != []

        if self.has_lindblad_noise:
            stacked = torch.stack(self.lindblad_ops)
            # The below is used for batch computation of noise collapse weights.
            self.aggregated_lindblad_ops = stacked.conj().transpose(1, 2) @ stacked

        self.lindblad_noise = compute_noise_from_lindbladians(self.lindblad_ops)
        self.jump_threshold = random.random()
        self.norm_gap_before_jump = self.state.norm() ** 2 - self.jump_threshold

    def init_initial_state(self, initial_state: State | None) -> None:
        if initial_state is None:
            well_prepared_qubits_count: int = (
                self.qubit_count
                if self.well_prepared_qubits_filter is None
                else sum(1 for x in self.well_prepared_qubits_filter if x)
            )

            self.state = MPS.make(
                well_prepared_qubits_count,
                precision=self.config.precision,
                max_bond_dim=self.config.max_bond_dim,
                num_gpus_to_use=self.config.num_gpus_to_use,
            )
            return

        if self.well_prepared_qubits_filter is not None:
            raise NotImplementedError(
                "Specifying the initial state in the presence "
                "of state preparation errors is currently not implemented."
            )

        assert isinstance(initial_state, MPS)
        initial_state = MPS(
            # Deep copy of every tensor of the initial state.
            [f.clone().detach() for f in initial_state.factors],
            precision=self.config.precision,
            max_bond_dim=self.config.max_bond_dim,
            num_gpus_to_use=self.config.num_gpus_to_use,
        )
        initial_state.truncate()
        initial_state *= 1 / initial_state.norm()
        self.state = initial_state

    def evolve_to_time(self, intermediate_time: float) -> float:
        """
        Internal method to evolve the state to `intermediate_time`
        with the FIXED current self.hamiltonian.
        Sets and returns the relative distance to the quantum jump threshold.
        """
        _TIME_CONVERSION_COEFF = 0.001  # Omega and delta are given in rad/ms, dt in ns
        delta_time = intermediate_time - self.current_time
        evolve_tdvp(
            t=-_TIME_CONVERSION_COEFF * delta_time * 1j,
            state=self.state,
            hamiltonian=self.hamiltonian,
            extra_krylov_tolerance=self.config.extra_krylov_tolerance,
            max_krylov_dim=self.config.max_krylov_dim,
            is_hermitian=not self.has_lindblad_noise,
        )
        self.current_time = intermediate_time
        self.norm_gap_before_jump = self.state.norm() ** 2 - self.jump_threshold
        return self.norm_gap_before_jump

    def do_time_step(self, step: int) -> None:
        """
        Updates hamiltonian and evolves state by dt.
        """
        interaction_matrix = (
            self.masked_interaction_matrix
            if self.current_time < self.slm_end_time
            else self.full_interaction_matrix
        )

        self.hamiltonian = make_H(
            interaction_matrix=interaction_matrix,
            omega=self.omega[step, :],
            delta=self.delta[step, :],
            phi=self.phi[step, :],
            hamiltonian_type=self.hamiltonian_type,
            noise=self.lindblad_noise,
            num_gpus_to_use=self.config.num_gpus_to_use,
        )

        target_time = float((step + 1) * self.config.dt)
        while self.current_time != target_time:
            assert self.current_time < target_time

            time_at_begin = self.current_time
            norm_gap_at_begin = self.norm_gap_before_jump
            assert not self.has_lindblad_noise or norm_gap_at_begin >= 0.0

            norm_gap_at_target_time = self.evolve_to_time(target_time)

            should_jump: bool = self.has_lindblad_noise and norm_gap_at_target_time < 0
            if should_jump:
                # Evolve to the intermediate time between self.current_time and target_time
                # where the jump should occur. That corresponds to self.evolve_to_time(t_jump) == 0.
                find_root_brents(
                    f=self.evolve_to_time,
                    start=time_at_begin,
                    f_start=norm_gap_at_begin,
                    end=target_time,
                    f_end=norm_gap_at_target_time,
                    tolerance=1.0,
                )
                self.do_random_quantum_jump()

        assert self.current_time == target_time

    def do_random_quantum_jump(self) -> None:
        jump_operator_weights = self.state.expect_batch(self.aggregated_lindblad_ops).real

        jumped_qubit_index, jump_operator = random.choices(
            [
                (qubit, op)
                for qubit in range(self.state.num_sites)
                for op in self.lindblad_ops
            ],
            weights=jump_operator_weights.reshape(-1),
        )[0]

        self.state.apply(jumped_qubit_index, jump_operator)
        self.state *= 1 / self.state.norm()

        norm_after_normalizing = self.state.norm()
        assert math.isclose(norm_after_normalizing, 1, abs_tol=1e-10)
        self.jump_threshold = random.uniform(0.0, norm_after_normalizing**2)
        self.norm_gap_before_jump = norm_after_normalizing**2 - self.jump_threshold

    def fill_results(self, results: Results, step: int) -> None:
        t = (step + 1) * self.config.dt  # we are now after the time-step, so use step+1

        noiseless_hamiltonian = (
            self.hamiltonian
            if not self.has_lindblad_noise
            else make_H(
                interaction_matrix=self.full_interaction_matrix,
                omega=self.omega[step, :],
                delta=self.delta[step, :],
                phi=self.phi[step, :],
                hamiltonian_type=self.hamiltonian_type,
                num_gpus_to_use=self.config.num_gpus_to_use
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

    def log_step_statistics(
        self, results: Results, *, step: int, duration: float
    ) -> None:
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

        if results.statistics is None:
            assert step == 0
            results.statistics = {"steps": []}

        assert "steps" in results.statistics
        assert len(results.statistics["steps"]) == step

        results.statistics["steps"].append(
            {
                "max_bond_dimension": self.state.get_max_bond_dim(),
                "memory_footprint": self.state.get_memory_footprint(),
                "RSS": max_mem,
                "duration": duration,
            }
        )
