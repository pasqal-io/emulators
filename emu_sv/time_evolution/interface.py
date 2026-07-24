from abc import ABC, abstractmethod
import torch
from emu_sv.rydberg_hamiltonian import RydbergHamiltonian
from emu_sv.lindblad_operator import RydbergLindbladian


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
