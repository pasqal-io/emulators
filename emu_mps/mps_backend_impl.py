import math
import os
import pathlib
import pickle
import random
import time
import typing
import uuid

from copy import deepcopy
from collections import Counter
from enum import Enum, auto
from types import MethodType
from typing import Any, Optional

import torch
from pulser import Sequence
from pulser.backend import EmulationConfig, Observable, Results, State

from emu_base import DEVICE_COUNT, PulserData, get_max_rss
from emu_base.math.brents_root_finding import BrentsRootFinder
from emu_base.utils import deallocate_tensor

from emu_mps.hamiltonian import make_H, update_H
from emu_mps.mpo import MPO
from emu_mps.mps import MPS
from emu_mps.mps_config import MPSConfig
from emu_base.jump_lindblad_operators import compute_noise_from_lindbladians
import emu_mps.optimatrix as optimat
from emu_mps.solver import Solver
from emu_mps.solver_utils import (
    evolve_pair,
    evolve_single,
    minimize_energy_pair,
    new_right_bath,
    right_baths,
)
from emu_mps.utils import (
    extended_mpo_factors,
    extended_mps_factors,
    get_extended_site_index,
    new_left_bath,
)


class Statistics(Observable):
    def __init__(
        self,
        evaluation_times: typing.Sequence[float] | None,
        data: list[float],
        timestep_count: int,
    ):
        super().__init__(evaluation_times=evaluation_times)
        self.data = data
        self.timestep_count = timestep_count

    @property
    def _base_tag(self) -> str:
        return "statistics"

    def apply(
        self,
        *,
        config: EmulationConfig,
        state: State,
        **kwargs: Any,
    ) -> dict:
        """Calculates the observable to store in the Results."""
        assert isinstance(state, MPS)
        duration = self.data[-1]
        max_mem = get_max_rss(state.factors[0].is_cuda)

        config.logger.info(
            f"step = {len(self.data)}/{self.timestep_count}, "
            + f"χ = {state.get_max_bond_dim()}, "
            + f"|ψ| = {state.get_memory_footprint():.3f} MB, "
            + f"RSS = {max_mem:.3f} MB, "
            + f"Δt = {duration:.3f} s"
        )

        return {
            "max_bond_dimension": state.get_max_bond_dim(),
            "memory_footprint": state.get_memory_footprint(),
            "RSS": max_mem,
            "duration": duration,
        }


class SwipeDirection(Enum):
    LEFT_TO_RIGHT = auto()
    RIGHT_TO_LEFT = auto()


class MPSBackendImpl:
    current_time: float = (
        0.0  # While dt is an integer, noisy collapse can happen at non-integer times.
    )
    well_prepared_qubits_filter: Optional[torch.Tensor]
    hamiltonian: MPO
    state: MPS
    right_baths: list[torch.Tensor]
    sweep_index: int
    swipe_direction: SwipeDirection
    timestep_index: int
    target_time: float
    results: Results

    def __init__(self, mps_config: MPSConfig, pulser_data: PulserData):
        self.config = mps_config
        self.target_times = pulser_data.target_times
        self.target_time = self.target_times[1]
        self.pulser_data = pulser_data
        self.qubit_count = pulser_data.qubit_count
        assert self.qubit_count >= 2
        self.omega = pulser_data.omega
        self.delta = pulser_data.delta
        self.phi = pulser_data.phi
        self.timestep_count: int = self.omega.shape[0]
        self.has_lindblad_noise = pulser_data.has_lindblad_noise
        self.lindblad_noise = torch.zeros(2, 2, dtype=torch.complex128)
        self.qubit_permutation = (
            optimat.minimize_bandwidth(pulser_data.full_interaction_matrix)
            if self.config.optimize_qubit_ordering
            else optimat.eye_permutation(self.qubit_count)
        )
        self.full_interaction_matrix = optimat.permute_tensor(
            pulser_data.full_interaction_matrix, self.qubit_permutation
        )
        self.masked_interaction_matrix = optimat.permute_tensor(
            pulser_data.masked_interaction_matrix, self.qubit_permutation
        )
        self.hamiltonian_type = pulser_data.hamiltonian_type
        self.slm_end_time = pulser_data.slm_end_time
        self.is_masked = self.slm_end_time > 0.0
        self.left_baths: list[torch.Tensor]
        self.time = time.time()
        self.swipe_direction = SwipeDirection.LEFT_TO_RIGHT
        self.sweep_index = 0
        self.timestep_index = 0
        self.results = Results(
            atom_order=optimat.permute_tuple(
                pulser_data.qubit_ids, self.qubit_permutation
            ),
            total_duration=self.target_times[-1],
        )
        self.statistics = Statistics(
            evaluation_times=[t / self.target_times[-1] for t in self.target_times],
            data=[],
            timestep_count=self.timestep_count,
        )
        self.autosave_file = self._get_autosave_filepath(self.config.autosave_prefix)
        self.config.logger.debug(
            f"""Will save simulation state to file "{self.autosave_file.name}"
            every {self.config.autosave_dt} seconds.\n"""
            f"""To resume: `MPSBackend().resume("{self.autosave_file}")`"""
        )
        self.last_save_time = time.time()

        if self.config.num_gpus_to_use > DEVICE_COUNT:
            self.config.logger.warning(
                f"Requested to use {self.config.num_gpus_to_use} GPU(s) "
                f"but only {DEVICE_COUNT if DEVICE_COUNT > 0 else 'cpu'} available"
            )

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        cp = deepcopy(self.config)
        d["config"] = cp
        d["state"].config = cp
        for obs in cp.observables:
            obs.apply = MethodType(type(obs).apply, obs)  # type: ignore[method-assign]
        # mypy thinks the method below is an attribute, because of the __getattr__ override
        d["results"] = self.results._to_abstract_repr()  # type: ignore[operator]
        return d

    def __setstate__(self, d: dict) -> None:
        self.__dict__ = d
        self.results = Results._from_abstract_repr(d["results"])  # type: ignore [attr-defined]
        self.config.monkeypatch_observables()

    @staticmethod
    def _get_autosave_filepath(autosave_prefix: str) -> pathlib.Path:
        return pathlib.Path(os.getcwd()) / (autosave_prefix + str(uuid.uuid1()) + ".dat")

    def init_dark_qubits(self) -> None:
        # has_state_preparation_error
        if self.config.noise_model.state_prep_error > 0.0:
            bad_atoms = self.pulser_data.hamiltonian.bad_atoms
            self.well_prepared_qubits_filter = torch.logical_not(
                torch.tensor(list(bool(x) for x in bad_atoms.values()))
            )
        else:
            self.well_prepared_qubits_filter = None

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
        if not torch.equal(
            self.qubit_permutation, optimat.eye_permutation(self.qubit_count)
        ):
            # permute the initial state to match with permuted Hamiltonian
            abstr_repr = initial_state._to_abstract_repr()
            eigs = abstr_repr["eigenstates"]
            ampl = {
                optimat.permute_string(bstr, self.qubit_permutation): amp
                for bstr, amp in abstr_repr["amplitudes"].items()
            }
            initial_state = MPS.from_state_amplitudes(eigenstates=eigs, amplitudes=ampl)

        initial_state = MPS(
            # Deep copy of every tensor of the initial state.
            [f.detach().clone() for f in initial_state.factors],
            precision=self.config.precision,
            max_bond_dim=self.config.max_bond_dim,
            num_gpus_to_use=self.config.num_gpus_to_use,
            eigenstates=initial_state.eigenstates,
        )
        initial_state.truncate()
        initial_state *= 1 / initial_state.norm()
        self.state = initial_state
        self.state.orthogonalize(0)

    def init_hamiltonian(self) -> None:
        """
        Must be called AFTER init_dark_qubits otherwise,
        too many factors are put in the Hamiltonian
        """
        self.hamiltonian = make_H(
            interaction_matrix=(
                self.masked_interaction_matrix
                if self.is_masked
                else self.full_interaction_matrix
            ),
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

    def get_current_right_bath(self) -> torch.Tensor:
        return self.right_baths[-1]

    def get_current_left_bath(self) -> torch.Tensor:
        return self.left_baths[-1]

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

        baths = (self.get_current_left_bath(), self.get_current_right_bath())

        if len(indices) == 1:
            assert orth_center_right is None
            (index,) = indices
            assert self.state.orthogonality_center == index

            self.state.factors[index] = evolve_single(
                state_factor=self.state.factors[index],
                ham_factor=self.hamiltonian.factors[index],
                baths=baths,
                dt=dt,
                config=self.config,
                is_hermitian=not self.has_lindblad_noise,
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
                config=self.config,
                orth_center_right=orth_center_right,
                is_hermitian=not self.has_lindblad_noise,
            )

            self.state.orthogonality_center = r if orth_center_right else l

    def progress(self) -> None:
        """
        Do one unit of simulation work given the current state.
        Update the state accordingly.
        The state of the simulation is stored in self.sweep_index and self.swipe_direction.
        """
        if self.is_finished():
            return

        delta_time = self.target_time - self.current_time

        assert self.qubit_count >= 1
        if 1 <= self.qubit_count <= 2:
            # Corner case: only 1 or 2 qubits
            assert self.swipe_direction == SwipeDirection.LEFT_TO_RIGHT
            assert self.sweep_index == 0

            if self.qubit_count == 1:
                self._evolve(0, dt=delta_time)
            else:
                self._evolve(0, 1, dt=delta_time, orth_center_right=False)

            self.sweep_complete()

        elif (
            self.sweep_index < self.qubit_count - 2
            and self.swipe_direction == SwipeDirection.LEFT_TO_RIGHT
        ):
            # Left-to-right swipe of TDVP
            self._evolve(
                self.sweep_index,
                self.sweep_index + 1,
                dt=delta_time / 2,
                orth_center_right=True,
            )
            self.left_baths.append(
                new_left_bath(
                    self.get_current_left_bath(),
                    self.state.factors[self.sweep_index],
                    self.hamiltonian.factors[self.sweep_index],
                ).to(self.state.factors[self.sweep_index + 1].device)
            )
            self._evolve(self.sweep_index + 1, dt=-delta_time / 2)
            self.right_baths.pop()
            self.sweep_index += 1

        elif (
            self.sweep_index == self.qubit_count - 2
            and self.swipe_direction == SwipeDirection.LEFT_TO_RIGHT
        ):
            # Time-evolution of the rightmost 2 tensors
            self._evolve(
                self.sweep_index,
                self.sweep_index + 1,
                dt=delta_time,
                orth_center_right=False,
            )
            self.swipe_direction = SwipeDirection.RIGHT_TO_LEFT

        elif (
            1 <= self.sweep_index and self.swipe_direction == SwipeDirection.RIGHT_TO_LEFT
        ):
            # Right-to-left swipe of TDVP
            assert self.sweep_index <= self.qubit_count - 2
            self.right_baths.append(
                new_right_bath(
                    self.get_current_right_bath(),
                    self.state.factors[self.sweep_index + 1],
                    self.hamiltonian.factors[self.sweep_index + 1],
                ).to(self.state.factors[self.sweep_index].device)
            )
            if not self.has_lindblad_noise:
                # Free memory because it won't be used anymore
                deallocate_tensor(self.right_baths[-2])

            self._evolve(self.sweep_index, dt=-delta_time / 2)
            self.left_baths.pop()

            self._evolve(
                self.sweep_index - 1,
                self.sweep_index,
                dt=delta_time / 2,
                orth_center_right=False,
            )
            self.sweep_index -= 1

            if self.sweep_index == 0:
                self.sweep_complete()
                self.swipe_direction = SwipeDirection.LEFT_TO_RIGHT

        else:
            raise Exception("Didn't expect this")

        self.save_simulation()

    def sweep_complete(self) -> None:
        self.current_time = self.target_time
        self.timestep_complete()

    def timestep_complete(self) -> None:
        self.fill_results()
        self.timestep_index += 1
        if self.is_masked and self.current_time >= self.slm_end_time:
            self.is_masked = False
            self.hamiltonian = make_H(
                interaction_matrix=self.full_interaction_matrix,
                hamiltonian_type=self.hamiltonian_type,
                num_gpus_to_use=self.config.num_gpus_to_use,
            )

        if not self.is_finished():
            self.target_time = self.target_times[self.timestep_index + 1]
            update_H(
                hamiltonian=self.hamiltonian,
                omega=self.omega[self.timestep_index, :],
                delta=self.delta[self.timestep_index, :],
                phi=self.phi[self.timestep_index, :],
                noise=self.lindblad_noise,
            )
            self.init_baths()

        self.statistics.data.append(time.time() - self.time)
        self.statistics(
            self.config,
            self.current_time / self.target_times[-1],
            self.state,
            self.hamiltonian,
            self.results,
        )
        self.time = time.time()

    def save_simulation(self) -> None:
        if self.last_save_time > time.time() - self.config.autosave_dt:
            return

        basename = self.autosave_file
        with open(basename.with_suffix(".new"), "wb") as file_handle:
            pickle.dump(self, file_handle)
        if basename.is_file():
            os.rename(basename, basename.with_suffix(".bak"))

        os.rename(basename.with_suffix(".new"), basename)
        autosave_filesize = os.path.getsize(self.autosave_file) / 1e6

        if basename.with_suffix(".bak").is_file():
            os.remove(basename.with_suffix(".bak"))

        self.last_save_time = time.time()

        self.config.logger.debug(
            f"Saved simulation state in file {self.autosave_file} ({autosave_filesize}MB)"
        )

    def fill_results(self) -> None:
        normalized_state = 1 / self.state.norm() * self.state

        current_time_int: int = round(self.current_time)
        fractional_time = self.current_time / self.target_times[-1]
        assert abs(self.current_time - current_time_int) < 1e-10

        if self.well_prepared_qubits_filter is None:
            for callback in self.config.observables:
                callback(
                    self.config,
                    fractional_time,
                    normalized_state,
                    self.hamiltonian,
                    self.results,
                )
            return

        full_mpo, full_state = None, None
        for callback in self.config.observables:
            time_tol = 0.5 / self.target_times[-1] + 1e-10
            if (
                callback.evaluation_times is not None
                and self.config.is_time_in_evaluation_times(
                    fractional_time, callback.evaluation_times, tol=time_tol
                )
            ) or self.config.is_evaluation_time(fractional_time, tol=time_tol):

                if full_mpo is None or full_state is None:
                    # Only do this potentially expensive step once and when needed.
                    full_mpo = MPO(
                        extended_mpo_factors(
                            self.hamiltonian.factors, self.well_prepared_qubits_filter
                        )
                    )
                    full_state = MPS(
                        extended_mps_factors(
                            normalized_state.factors,
                            self.well_prepared_qubits_filter,
                        ),
                        num_gpus_to_use=None,  # Keep the already assigned devices.
                        orthogonality_center=get_extended_site_index(
                            self.well_prepared_qubits_filter,
                            normalized_state.orthogonality_center,
                        ),
                        eigenstates=normalized_state.eigenstates,
                    )

                callback(self.config, fractional_time, full_state, full_mpo, self.results)

    def permute_results(self, results: Results, permute: bool) -> Results:
        if permute:
            inv_perm = optimat.inv_permutation(self.qubit_permutation)
            permute_bitstrings(results, inv_perm)
            permute_occupations_and_correlations(results, inv_perm)
            permute_atom_order(results, inv_perm)
        return results


def permute_bitstrings(results: Results, perm: torch.Tensor) -> None:
    if "bitstrings" not in results.get_result_tags():
        return
    uuid_bs = results._find_uuid("bitstrings")

    results._results[uuid_bs] = [
        Counter({optimat.permute_string(bstr, perm): c for bstr, c in bs_counter.items()})
        for bs_counter in results._results[uuid_bs]
    ]


def permute_occupations_and_correlations(results: Results, perm: torch.Tensor) -> None:
    for corr in ["occupation", "correlation_matrix"]:
        if corr not in results.get_result_tags():
            continue

        uuid_corr = results._find_uuid(corr)
        corrs = results._results[uuid_corr]
        results._results[uuid_corr] = (
            [  # vector quantities become lists after results are serialized (e.g. for checkpoints)
                optimat.permute_tensor(
                    corr if isinstance(corr, torch.Tensor) else torch.tensor(corr), perm
                )
                for corr in corrs
            ]
        )


def permute_atom_order(results: Results, perm: torch.Tensor) -> None:
    at_ord = list(results.atom_order)
    at_ord = optimat.permute_list(at_ord, perm)
    results.atom_order = tuple(at_ord)


class NoisyMPSBackendImpl(MPSBackendImpl):
    """
    Version of MPSBackendImpl with non-zero lindbladian noise.
    Implements the Monte-Carlo Wave Function jump method.
    """

    jump_threshold: float
    aggregated_lindblad_ops: torch.Tensor
    norm_gap_before_jump: float
    root_finder: Optional[BrentsRootFinder]

    def __init__(self, config: MPSConfig, pulser_data: PulserData):
        super().__init__(config, pulser_data)
        self.lindblad_ops = pulser_data.lindblad_ops
        self.root_finder = None

        assert self.has_lindblad_noise

    def init_lindblad_noise(self) -> None:
        stacked = torch.stack(self.lindblad_ops)
        # The below is used for batch computation of noise collapse weights.
        self.aggregated_lindblad_ops = stacked.conj().transpose(1, 2) @ stacked

        self.lindblad_noise = compute_noise_from_lindbladians(self.lindblad_ops)

    def set_jump_threshold(self, bound: float) -> None:
        self.jump_threshold = random.uniform(0.0, bound)
        self.norm_gap_before_jump = self.state.norm().item() ** 2 - self.jump_threshold

    def init(self) -> None:
        self.init_lindblad_noise()
        super().init()
        self.set_jump_threshold(1.0)

    def sweep_complete(self) -> None:
        previous_time = self.current_time
        self.current_time = self.target_time
        previous_norm_gap_before_jump = self.norm_gap_before_jump
        self.norm_gap_before_jump = self.state.norm().item() ** 2 - self.jump_threshold

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

        self.norm_gap_before_jump = self.state.norm().item() ** 2 - self.jump_threshold
        self.root_finder.provide_ordinate(self.current_time, self.norm_gap_before_jump)

        if self.root_finder.is_converged(tolerance=1):
            self.do_random_quantum_jump()
            self.target_time = self.target_times[self.timestep_index + 1]
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
            weights=jump_operator_weights.view(-1).tolist(),
        )[0]

        self.state.apply(jumped_qubit_index, jump_operator)
        self.state.orthogonalize(0)
        self.state *= 1 / self.state.norm()
        self.init_baths()

        norm_after_normalizing = self.state.norm().item()
        assert math.isclose(norm_after_normalizing, 1, abs_tol=1e-10)
        self.set_jump_threshold(norm_after_normalizing**2)

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


class DMRGBackendImpl(MPSBackendImpl):
    def __init__(
        self,
        mps_config: MPSConfig,
        pulser_data: PulserData,
        energy_tolerance: float = 1e-5,
        max_sweeps: int = 2000,
    ):

        if mps_config.noise_model.noise_types != ():
            raise NotImplementedError(
                "DMRG solver does not currently support noise types"
                f"you are using: {mps_config.noise_model.noise_types}"
            )
        super().__init__(mps_config, pulser_data)
        self.previous_energy: Optional[float] = None
        self.current_energy: Optional[float] = None
        self.sweep_count: int = 0
        self.energy_tolerance: float = energy_tolerance
        self.max_sweeps: int = max_sweeps

    def convergence_check(self, energy_tolerance: float) -> bool:
        if self.previous_energy is None or self.current_energy is None:
            return False
        return abs(self.current_energy - self.previous_energy) < energy_tolerance

    def progress(self) -> None:
        if self.is_finished():
            return

        # perform one two-site energy minimization and update
        idx = self.sweep_index
        assert self.swipe_direction in (
            SwipeDirection.LEFT_TO_RIGHT,
            SwipeDirection.RIGHT_TO_LEFT,
        ), "Unknown Swipe direction"

        orth_center_right = self.swipe_direction == SwipeDirection.LEFT_TO_RIGHT
        new_L, new_R, energy = minimize_energy_pair(
            state_factors=self.state.factors[idx : idx + 2],
            ham_factors=self.hamiltonian.factors[idx : idx + 2],
            baths=(self.left_baths[-1], self.right_baths[-1]),
            orth_center_right=orth_center_right,
            config=self.config,
            residual_tolerance=self.config.precision,
        )
        self.state.factors[idx], self.state.factors[idx + 1] = new_L, new_R
        self.state.orthogonality_center = idx + 1 if orth_center_right else idx
        self.current_energy = energy

        # updating baths and orthogonality center
        if self.swipe_direction == SwipeDirection.LEFT_TO_RIGHT:
            self._left_to_right_update(idx)
        elif self.swipe_direction == SwipeDirection.RIGHT_TO_LEFT:
            self._right_to_left_update(idx)
        else:
            raise Exception("Did not expect this")

        self.save_simulation()

    def _left_to_right_update(self, idx: int) -> None:
        if idx < self.qubit_count - 2:
            self.left_baths.append(
                new_left_bath(
                    self.get_current_left_bath(),
                    self.state.factors[idx],
                    self.hamiltonian.factors[idx],
                ).to(self.state.factors[idx + 1].device)
            )
            self.right_baths.pop()
            self.sweep_index += 1

        if self.sweep_index == self.qubit_count - 2:
            self.swipe_direction = SwipeDirection.RIGHT_TO_LEFT

    def _right_to_left_update(self, idx: int) -> None:
        if idx > 0:
            self.right_baths.append(
                new_right_bath(
                    self.get_current_right_bath(),
                    self.state.factors[idx + 1],
                    self.hamiltonian.factors[idx + 1],
                ).to(self.state.factors[idx].device)
            )
            self.left_baths.pop()
            self.sweep_index -= 1

        if self.sweep_index == 0:
            self.state.orthogonalize(0)
            self.swipe_direction = SwipeDirection.LEFT_TO_RIGHT
            self.sweep_count += 1
            self.sweep_complete()

    def sweep_complete(self) -> None:
        # This marks the end of one full sweep: checking convergence
        if self.convergence_check(self.energy_tolerance):
            self.current_time = self.target_time
            self.timestep_complete()
        elif self.sweep_count + 1 > self.max_sweeps:
            # not converged
            raise RuntimeError(f"DMRG did not converge after {self.max_sweeps} sweeps")
        else:
            # not converged for the current sweep. restart
            self.previous_energy = self.current_energy

        assert self.sweep_index == 0
        assert self.state.orthogonality_center == 0
        assert self.swipe_direction == SwipeDirection.LEFT_TO_RIGHT
        self.current_energy = None


def create_impl(sequence: Sequence, config: MPSConfig) -> MPSBackendImpl:
    pulser_data = PulserData(sequence=sequence, config=config, dt=config.dt)

    if pulser_data.has_lindblad_noise:
        return NoisyMPSBackendImpl(config, pulser_data)
    if config.solver == Solver.DMRG:
        return DMRGBackendImpl(config, pulser_data)
    return MPSBackendImpl(config, pulser_data)
