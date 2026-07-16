import torch
import random
import math
from abc import ABC, abstractmethod
from typing import cast
from emu_base.math.krylov_exp import krylov_exp
from emu_base.math.brents_root_finding import BrentsRootFinder
from emu_base.jump_lindblad_operators import compute_noise_from_lindbladians
from emu_sv.hamiltonian import RydbergHamiltonian
from emu_sv.lindblad_operator import RydbergLindbladian
from emu_sv.algebra import apply, expect_batch


class BaseStepper(ABC):
    @staticmethod
    @abstractmethod
    def get_hamiltonian(
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        pulser_lindblads: list[torch.Tensor],
        interaction_matrix: torch.Tensor,
        device: torch.device,
    ) -> RydbergLindbladian | RydbergHamiltonian:
        pass

    @abstractmethod
    def apply(
        self,
        dt: float,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        full_interaction_matrix: torch.Tensor,
        state: torch.Tensor,
        krylov_tolerance: float,
        pulser_lindblads: list[torch.Tensor],
    ) -> (
        tuple[torch.Tensor, RydbergLindbladian] | tuple[torch.Tensor, RydbergHamiltonian]
    ):
        pass


class EvolveDensityMatrix(BaseStepper):
    """Evolution of a density matrix under a Lindbladian operator."""

    @staticmethod
    def get_hamiltonian(
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        pulser_lindblads: list[torch.Tensor],
        interaction_matrix: torch.Tensor,
        device: torch.device,
    ) -> RydbergLindbladian:
        return RydbergLindbladian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            pulser_lindblads=pulser_lindblads,
            interaction_matrix=interaction_matrix,
            device=device,
        )

    def apply(
        self,
        dt: float,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        full_interaction_matrix: torch.Tensor,
        state: torch.Tensor,
        krylov_tolerance: float,
        pulser_lindblads: list[torch.Tensor],
    ) -> tuple[torch.Tensor, RydbergLindbladian]:
        ham = self.get_hamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            pulser_lindblads=pulser_lindblads,
            interaction_matrix=full_interaction_matrix,
            device=state.device,
        )

        def op(x: torch.Tensor) -> torch.Tensor:
            return -1j * dt * (ham @ x)

        return (
            krylov_exp(
                op,
                state,
                norm_tolerance=krylov_tolerance,
                exp_tolerance=krylov_tolerance,
                is_hermitian=False,
            ),
            ham,
        )


class EvolveMonteCarlo(BaseStepper):
    """Evolution of a state vector under Monte Carlo quantum jumps."""

    def __init__(self) -> None:
        self.jump_threshold = random.uniform(0.0, 1.0)

    @staticmethod
    def get_hamiltonian(
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        pulser_lindblads: list[torch.Tensor],
        interaction_matrix: torch.Tensor,
        device: torch.device,
    ) -> RydbergHamiltonian:
        return RydbergHamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            interaction_matrix=interaction_matrix,
            device=device,
            noise=compute_noise_from_lindbladians(pulser_lindblads),
        )

    @staticmethod
    def evolve(
        dt: float,
        hamiltonian: RydbergHamiltonian,
        state: torch.Tensor,
        krylov_tolerance: float,
    ) -> torch.Tensor:

        def op(x: torch.Tensor) -> torch.Tensor:
            return -1j * dt * (hamiltonian * x)

        return krylov_exp(
            op,
            state,
            norm_tolerance=krylov_tolerance,
            exp_tolerance=krylov_tolerance,
            is_hermitian=False,
        )

    def do_quantum_jump(
        self,
        state: torch.Tensor,
        lindblad_ops: list[torch.Tensor],
        aggregated_lindblad_ops: torch.Tensor,
        n_qubits: int,
    ) -> torch.Tensor:
        jump_operator_weights = expect_batch(
            state, aggregated_lindblad_ops, n_qubits
        ).real
        jumped_qubit_index, jump_operator = random.choices(
            [(qubit, op) for qubit in range(n_qubits) for op in lindblad_ops],
            weights=jump_operator_weights.view(-1).tolist(),
        )[0]

        state = apply(state, jumped_qubit_index, jump_operator)

        state *= 1 / torch.linalg.vector_norm(state)

        norm_after_normalizing = torch.linalg.vector_norm(state).item()

        assert math.isclose(norm_after_normalizing, 1, abs_tol=1e-10)
        self.jump_threshold = random.uniform(0.0, norm_after_normalizing**2)
        return cast(torch.Tensor, state)

    def apply(
        self,
        dt: float,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        full_interaction_matrix: torch.Tensor,
        state: torch.Tensor,
        krylov_tolerance: float,
        pulser_lindblads: list[torch.Tensor],
    ) -> tuple[torch.Tensor, RydbergHamiltonian]:
        ham = self.get_hamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            pulser_lindblads=pulser_lindblads,
            interaction_matrix=full_interaction_matrix,
            device=state.device,
        )

        current_time = 0.0
        tol = dt / 10
        new_norm_gap = torch.linalg.vector_norm(state) ** 2 - self.jump_threshold

        while abs(dt - current_time) > 1e-5:
            old_norm_gap = new_norm_gap
            state = self.evolve(dt - current_time, ham, state, krylov_tolerance)

            new_norm_gap = torch.linalg.vector_norm(state) ** 2 - self.jump_threshold
            if new_norm_gap > 0.0:
                break
            else:
                root_finder = BrentsRootFinder(
                    start=current_time,
                    end=dt,
                    f_start=old_norm_gap,
                    f_end=new_norm_gap,
                    epsilon=tol,
                )
                current_time = dt
                while not root_finder.is_converged(tolerance=tol):
                    target_time = root_finder.get_next_abscissa()
                    state = self.evolve(
                        target_time - current_time, ham, state, krylov_tolerance
                    )
                    current_time = target_time
                    new_norm_gap = (
                        torch.linalg.vector_norm(state) ** 2 - self.jump_threshold
                    )
                    root_finder.provide_ordinate(current_time, new_norm_gap)

                stacked = torch.stack(pulser_lindblads)
                # The below is used for batch computation of noise collapse weights.
                aggregated_lindblad_ops = stacked.conj().transpose(1, 2) @ stacked
                state = self.do_quantum_jump(
                    state, pulser_lindblads, aggregated_lindblad_ops, len(omegas)
                )
                new_norm_gap = torch.linalg.vector_norm(state) ** 2 - self.jump_threshold
        return state, ham
