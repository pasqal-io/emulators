from resource import RUSAGE_SELF, getrusage
from time import time
from typing import Optional

import torch
from pulser import Sequence

from emu_mps.base_classes.backend import Backend
from emu_mps.base_classes.config import BackendConfig
from emu_mps.base_classes.results import Results
from emu_mps.hamiltonian import make_H, rydberg_interaction
from emu_mps.mpo import MPO
from emu_mps.mps import MPS
from emu_mps.mps_config import MPSConfig
from emu_mps.noise import compute_noise_from_lindbladians, pick_well_prepared_qubits
from emu_mps.pulser_adapter import (
    extract_omega_delta_phi,
    get_all_lindblad_noise_operators,
)
from emu_mps.tdvp import evolve_tdvp
from emu_mps.utils import extended_mpo_factors, extended_mps_factors


class _RunImpl:
    well_prepared_qubits_filter: Optional[list[bool]]
    hamiltonian: MPO
    lindblad_ops: list[torch.Tensor]
    lindblad_noise: torch.Tensor

    def __init__(self, sequence: Sequence, mps_config: MPSConfig):
        self.sequence = sequence
        self.config = mps_config

        self.dt = self.config.dt
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

        # Work in progress.
        if self.lindblad_ops:
            raise NotImplementedError("Lindbladian noise is not supported yet")

        self.lindblad_noise = compute_noise_from_lindbladians(self.lindblad_ops)

    def make_initial_state(self) -> MPS:
        if self.config.initial_state is None:
            well_prepared_qubits_count: int = (
                self.qubit_count
                if self.well_prepared_qubits_filter is None
                else sum(1 for x in self.well_prepared_qubits_filter if x)
            )

            return MPS(
                well_prepared_qubits_count,
                truncate=False,
                precision=self.config.precision,
                max_bond_dim=self.config.max_bond_dim,
                num_devices_to_use=self.config.num_devices_to_use,
            )
        else:
            if self.well_prepared_qubits_filter is not None:
                raise NotImplementedError(
                    "Specifying the initial state in the presence of state \
                        preparation errors is currently not implemented."
                )
            assert isinstance(self.config.initial_state, MPS)
            return MPS(
                self.config.initial_state.factors,
                truncate=True,
                precision=self.config.precision,
                max_bond_dim=self.config.max_bond_dim,
                num_devices_to_use=self.config.num_devices_to_use,
            )

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
        evolve_tdvp(
            -_TIME_CONVERSION_COEFF * self.dt * 1j,
            self.state,
            self.hamiltonian,
            self.config.extra_krylov_tolerance,
            self.config.max_krylov_dim,
        )

    def fill_results(self, results: Results, step: int) -> None:
        t = (step + 1) * self.dt  # we are now after the time-step, so use step+1
        for callback in self.config.callbacks:
            if self.well_prepared_qubits_filter is None:
                callback(self.config, t, self.state, self.hamiltonian, results)
            elif t in callback.evaluation_times:
                full_mpo = MPO(
                    extended_mpo_factors(
                        self.hamiltonian.factors, self.well_prepared_qubits_filter
                    )
                )
                full_state = MPS(
                    extended_mps_factors(
                        self.state.factors, self.well_prepared_qubits_filter
                    ),
                    keep_devices=True,
                )
                callback(self.config, t, full_state, full_mpo, results)

    def print_step_statistics(self, *, step: int, duration: float) -> None:
        mem = (
            torch.cuda.max_memory_allocated() * 1e-6
            if self.state.factors[0].is_cuda
            else getrusage(RUSAGE_SELF).ru_maxrss * 1e-3
        )
        print(
            f"step = {step + 1}/{self.timestep_count},",
            f"χ = {self.state.get_max_bond_dim()},",
            f"|ψ| = {self.state.get_memory_footprint():.3f} MB,",
            f"RSS = {mem:.3f} MB,",
            f"Δt = {duration:.3f} s",
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

            impl.print_step_statistics(step=step, duration=end - start)

        return results
