import math
import random
from resource import RUSAGE_SELF, getrusage
from typing import Optional

import torch
import time
from pulser import Sequence

from emu_base import Results, State, PulserData
from emu_base.math.brents_root_finding import BrentsRootFinder
from emu_mps.hamiltonian import make_H, update_H
from emu_mps.mpo import MPO
from emu_mps.mps import MPS
from emu_mps.mps_config import MPSConfig
from emu_mps.noise import compute_noise_from_lindbladians, pick_well_prepared_qubits
from emu_mps.tdvp import (
    evolve_single,
    evolve_pair,
    EvolveConfig,
    new_right_bath,
    right_baths,
)
from emu_mps.utils import (
    extended_mpo_factors,
    extended_mps_factors,
    get_extended_site_index,
    new_left_bath,
)
from enum import Enum, auto


class SwipeDirection(Enum):
    LEFT_TO_RIGHT = auto()
    RIGHT_TO_LEFT = auto()


class MPSBackendImpl:
    current_time: float = (
        0.0  # While dt is an integer, noisy collapse can happen at non-integer times.
    )
    well_prepared_qubits_filter: Optional[list[bool]]
    hamiltonian: MPO
    state: MPS
    right_baths: list[torch.Tensor]
    tdvp_index: int
    swipe_direction: SwipeDirection
    timestep_index: int
    target_time: float
    results: Results

    def __init__(self, mps_config: MPSConfig, pulser_data: PulserData):
        self.config = mps_config
        self.target_time = float(self.config.dt)
        self.qubit_count = pulser_data.qubit_count
        assert self.qubit_count >= 2
        self.omega = pulser_data.omega
        self.delta = pulser_data.delta
        self.phi = pulser_data.phi
        self.timestep_count: int = self.omega.shape[0]
        self.has_lindblad_noise = pulser_data.has_lindblad_noise
        self.lindblad_noise = torch.zeros(2, 2, dtype=torch.complex128)
        self.full_interaction_matrix = pulser_data.full_interaction_matrix
        self.masked_interaction_matrix = pulser_data.masked_interaction_matrix
        self.hamiltonian_type = pulser_data.hamiltonian_type
        self.slm_end_time = pulser_data.slm_end_time
        self.is_masked = self.slm_end_time > 0.0
        self.left_baths: list[torch.Tensor]
        self.time = time.time()
        self.swipe_direction = SwipeDirection.LEFT_TO_RIGHT
        self.tdvp_index = 0
        self.timestep_index = 0
        self.results = Results()

        self.evolve_config = EvolveConfig(
            exp_tolerance=self.config.precision * self.config.extra_krylov_tolerance,
            norm_tolerance=self.config.precision * self.config.extra_krylov_tolerance,
            max_krylov_dim=self.config.max_krylov_dim,
            is_hermitian=not self.has_lindblad_noise,
            max_error=self.config.precision,
            max_rank=self.config.max_bond_dim,
        )

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
            self.qubit_count = sum(1 for x in self.well_prepared_qubits_filter if x)

            self.full_interaction_matrix = self.full_interaction_matrix[
                self.well_prepared_qubits_filter, :
            ][:, self.well_prepared_qubits_filter]
            self.masked_interaction_matrix = self.masked_interaction_matrix[
                self.well_prepared_qubits_filter, :
            ][:, self.well_prepared_qubits_filter]
            self.omega = self.omega[:, self.well_prepared_qubits_filter]
            self.delta = self.delta[:, self.well_prepared_qubits_filter]
            self.phi = self.phi[:, self.well_prepared_qubits_filter]

    def init_initial_state(self, initial_state: State | None = None) -> None:
        if initial_state is None:
            self.state = MPS.make(
                self.qubit_count,
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

    def init_hamiltonian(self) -> None:
        """
        Must be called AFTER init_dark_qubits otherwise,
        too many factors are put in the Hamiltonian
        """
        self.hamiltonian = make_H(
            interaction_matrix=self.masked_interaction_matrix
            if self.is_masked
            else self.full_interaction_matrix,
            hamiltonian_type=self.hamiltonian_type,
            num_gpus_to_use=self.config.num_gpus_to_use,
        )

        update_H(
            hamiltonian=self.hamiltonian,
            omega=self.omega[self.timestep_index, :],
            delta=self.delta[self.timestep_index, :],
            phi=self.phi[self.timestep_index, :],
            noise=self.lindblad_noise,
        )

    def init_baths(self) -> None:
        self.left_baths = [
            torch.ones(
                1, 1, 1, dtype=torch.complex128, device=self.state.factors[0].device
            )
        ]
        self.right_baths = right_baths(self.state, self.hamiltonian, final_qubit=2)
        assert len(self.right_baths) == self.qubit_count - 1

    def init(self) -> None:
        self.init_dark_qubits()
        self.init_initial_state(self.config.initial_state)
        self.init_hamiltonian()
        self.init_baths()

    def is_finished(self) -> bool:
        return self.timestep_index >= self.timestep_count

    def _evolve(
        self, *indices: int, dt: float, orth_center_right: Optional[bool] = None
    ) -> None:
        """
        Time-evolve the state's tensors located at the given 1 or 2 indices by dt,
        using the baths stored in self.left_baths and self.right_baths.
        When 2 indices are given, they need to be consecutive.
        Updates the state's orthogonality center according to orth_center_right.
        """
        assert 1 <= len(indices) <= 2

        baths = (self.left_baths[-1], self.right_baths[-1])

        if len(indices) == 1:
            assert orth_center_right is None
            (index,) = indices
            assert self.state.orthogonality_center == index

            self.state.factors[index] = evolve_single(
                state_factor=self.state.factors[index],
                ham_factor=self.hamiltonian.factors[index],
                baths=baths,
                dt=dt,
                config=self.evolve_config,
            )
        else:
            assert orth_center_right is not None
            l, r = indices
            assert r == l + 1, "Indices need to be consecutive"
            assert self.state.orthogonality_center in {l, r}, (
                "State needs to be orthogonalized" " on one of the evolved indices"
            )

            self.state.factors[l : r + 1] = evolve_pair(
                state_factors=self.state.factors[l : r + 1],
                ham_factors=self.hamiltonian.factors[l : r + 1],
                baths=baths,
                dt=dt,
                config=self.evolve_config,
                orth_center_right=orth_center_right,
            )

            self.state.orthogonality_center = r if orth_center_right else l

    def progress(self) -> None:
        """
        Do one unit of simulation work given the current state.
        Update the state accordingly.
        The state of the simulation is stored in self.tdvp_index and self.swipe_direction.
        """
        if self.is_finished():
            return

        delta_time = self.target_time - self.current_time

        assert self.qubit_count >= 1
        if 1 <= self.qubit_count <= 2:
            # Corner case: only 1 or 2 qubits
            assert self.swipe_direction == SwipeDirection.LEFT_TO_RIGHT
            assert self.tdvp_index == 0

            if self.qubit_count == 1:
                self._evolve(0, dt=delta_time)
            else:
                self._evolve(0, 1, dt=delta_time, orth_center_right=False)

            self.tdvp_complete()

        elif (
            self.tdvp_index < self.qubit_count - 2
            and self.swipe_direction == SwipeDirection.LEFT_TO_RIGHT
        ):
            # Left-to-right swipe of TDVP
            self._evolve(
                self.tdvp_index,
                self.tdvp_index + 1,
                dt=delta_time / 2,
                orth_center_right=True,
            )
            self.left_baths.append(
                new_left_bath(
                    self.left_baths[-1],
                    self.state.factors[self.tdvp_index],
                    self.hamiltonian.factors[self.tdvp_index],
                )
            )
            self._evolve(self.tdvp_index + 1, dt=-delta_time / 2)
            self.right_baths.pop()
            self.tdvp_index += 1

        elif (
            self.tdvp_index == self.qubit_count - 2
            and self.swipe_direction == SwipeDirection.LEFT_TO_RIGHT
        ):
            # Time-evolution of the rightmost 2 tensors
            self._evolve(
                self.tdvp_index,
                self.tdvp_index + 1,
                dt=delta_time,
                orth_center_right=False,
            )
            self.swipe_direction = SwipeDirection.RIGHT_TO_LEFT

        elif (
            1 <= self.tdvp_index and self.swipe_direction == SwipeDirection.RIGHT_TO_LEFT
        ):
            # Right-to-left swipe of TDVP
            assert self.tdvp_index <= self.qubit_count - 2
            self.right_baths.append(
                new_right_bath(
                    self.right_baths[-1],
                    self.state.factors[self.tdvp_index + 1],
                    self.hamiltonian.factors[self.tdvp_index + 1],
                )
            )
            if not self.has_lindblad_noise:
                # Free memory because it won't be used anymore
                self.right_baths[-2] = None

            self._evolve(self.tdvp_index, dt=-delta_time / 2)
            self.left_baths.pop()

            self._evolve(
                self.tdvp_index - 1,
                self.tdvp_index,
                dt=delta_time / 2,
                orth_center_right=False,
            )
            self.tdvp_index -= 1

            if self.tdvp_index == 0:
                self.tdvp_complete()
                self.swipe_direction = SwipeDirection.LEFT_TO_RIGHT

        else:
            raise Exception("Didn't expect this")

        # TODO: checkpoint/autosave here

    def tdvp_complete(self) -> None:
        self.current_time = self.target_time
        self.timestep_complete()

    def timestep_complete(self) -> None:
        self.fill_results()
        self.timestep_index += 1
        self.target_time = float((self.timestep_index + 1) * self.config.dt)
        if self.is_masked and self.current_time >= self.slm_end_time:
            self.is_masked = False
            self.hamiltonian = make_H(
                interaction_matrix=self.full_interaction_matrix,
                hamiltonian_type=self.hamiltonian_type,
                num_gpus_to_use=self.config.num_gpus_to_use,
            )

        if not self.is_finished():
            update_H(
                hamiltonian=self.hamiltonian,
                omega=self.omega[self.timestep_index, :],
                delta=self.delta[self.timestep_index, :],
                phi=self.phi[self.timestep_index, :],
                noise=self.lindblad_noise,
            )
            self.init_baths()

        self.log_step_statistics(duration=time.time() - self.time)
        self.time = time.time()

    def fill_results(self) -> None:
        normalized_state = 1 / self.state.norm() * self.state

        if self.well_prepared_qubits_filter is None:
            for callback in self.config.callbacks:
                callback(
                    self.config,
                    self.current_time,
                    normalized_state,
                    self.hamiltonian,
                    self.results,
                )
            return

        full_mpo, full_state = None, None
        for callback in self.config.callbacks:
            if self.current_time not in callback.evaluation_times:
                continue

            if full_mpo is None or full_state is None:
                # Only do this potentially expensive step once and when needed.
                full_mpo = MPO(
                    extended_mpo_factors(
                        self.hamiltonian.factors, self.well_prepared_qubits_filter
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

            callback(self.config, self.current_time, full_state, full_mpo, self.results)

    def log_step_statistics(self, *, duration: float) -> None:
        if self.state.factors[0].is_cuda:
            max_mem_per_device = (
                torch.cuda.max_memory_allocated(device) * 1e-6
                for device in range(torch.cuda.device_count())
            )
            max_mem = max(max_mem_per_device)
        else:
            max_mem = getrusage(RUSAGE_SELF).ru_maxrss * 1e-3

        self.config.logger.info(
            f"step = {self.timestep_index}/{self.timestep_count}, "
            + f"χ = {self.state.get_max_bond_dim()}, "
            + f"|ψ| = {self.state.get_memory_footprint():.3f} MB, "
            + f"RSS = {max_mem:.3f} MB, "
            + f"Δt = {duration:.3f} s"
        )

        if self.results.statistics is None:
            assert self.timestep_index == 1
            self.results.statistics = {"steps": []}

        assert "steps" in self.results.statistics
        assert len(self.results.statistics["steps"]) == self.timestep_index - 1

        self.results.statistics["steps"].append(
            {
                "max_bond_dimension": self.state.get_max_bond_dim(),
                "memory_footprint": self.state.get_memory_footprint(),
                "RSS": max_mem,
                "duration": duration,
            }
        )


class NoisyMPSBackendImpl(MPSBackendImpl):
    """
    Version of MPSBackendImpl with non-zero lindbladian noise.
    Implements the Monte-Carlo Wave Function jump method.
    """

    jump_threshold: float
    aggregated_lindblad_ops: Optional[torch.Tensor]
    norm_gap_before_jump: float
    root_finder: Optional[BrentsRootFinder]

    def __init__(self, config: MPSConfig, pulser_data: PulserData):
        super().__init__(config, pulser_data)
        self.aggregated_lindblad_ops = None
        self.lindblad_ops = pulser_data.lindblad_ops
        self.root_finder = None

        assert self.has_lindblad_noise

    def init_lindblad_noise(self) -> None:
        stacked = torch.stack(self.lindblad_ops)
        # The below is used for batch computation of noise collapse weights.
        self.aggregated_lindblad_ops = stacked.conj().transpose(1, 2) @ stacked

        self.lindblad_noise = compute_noise_from_lindbladians(self.lindblad_ops)
        self.jump_threshold = random.random()
        self.norm_gap_before_jump = self.state.norm() ** 2 - self.jump_threshold

    def init(self) -> None:
        super().init()
        self.init_lindblad_noise()

    def tdvp_complete(self) -> None:
        previous_time = self.current_time
        self.current_time = self.target_time
        previous_norm_gap_before_jump = self.norm_gap_before_jump
        self.norm_gap_before_jump = self.state.norm() ** 2 - self.jump_threshold

        if self.root_finder is None:
            # No quantum jump location finding in progress
            if self.norm_gap_before_jump < 0:
                # Initiate quantum jump location finding
                # Jump occurs when norm_gap_before_jump ~ 0
                self.root_finder = BrentsRootFinder(
                    start=previous_time,
                    end=self.current_time,
                    f_start=previous_norm_gap_before_jump,
                    f_end=self.norm_gap_before_jump,
                    epsilon=1,
                )
                self.target_time = self.root_finder.get_next_abscissa()
            else:
                self.timestep_complete()

            return

        self.norm_gap_before_jump = self.state.norm() ** 2 - self.jump_threshold
        self.root_finder.provide_ordinate(self.current_time, self.norm_gap_before_jump)

        if self.root_finder.is_converged(tolerance=1):
            self.do_random_quantum_jump()
            self.target_time = (self.timestep_index + 1) * self.config.dt
            self.root_finder = None
        else:
            self.target_time = self.root_finder.get_next_abscissa()

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
        self.state.orthogonalize(0)
        self.state *= 1 / self.state.norm()

        norm_after_normalizing = self.state.norm()
        assert math.isclose(norm_after_normalizing, 1, abs_tol=1e-10)
        self.jump_threshold = random.uniform(0.0, norm_after_normalizing**2)
        self.norm_gap_before_jump = norm_after_normalizing**2 - self.jump_threshold

    def fill_results(self) -> None:
        # Remove the noise from self.hamiltonian for the callbacks.
        # Since update_H is called at the start of do_time_step this is safe.
        update_H(
            hamiltonian=self.hamiltonian,
            omega=self.omega[self.timestep_index - 1, :],  # Meh
            delta=self.delta[self.timestep_index - 1, :],
            phi=self.phi[self.timestep_index - 1, :],
        )

        super().fill_results()


def create_impl(sequence: Sequence, config: MPSConfig) -> MPSBackendImpl:
    pulser_data = PulserData(sequence=sequence, config=config, dt=config.dt)

    if pulser_data.has_lindblad_noise:
        return NoisyMPSBackendImpl(config, pulser_data)

    return MPSBackendImpl(config, pulser_data)
