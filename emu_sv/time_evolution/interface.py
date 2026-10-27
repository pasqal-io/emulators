from abc import ABC, abstractmethod
import torch
from emu_sv.hamiltonian import RydbergHamiltonian, make_interaction_diagonal
from emu_sv.lindblad_operator import RydbergLindbladian


class BaseStepper(ABC):
    # (interaction matrix, interaction diagonal) of the last time step
    _interaction: tuple[torch.Tensor, torch.Tensor] | None = None

    def __init__(self, max_krylov_dim: int) -> None:
        self.max_krylov_dim = max_krylov_dim

    def interaction_diagonal(
        self, interaction_matrix: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Return the interaction part ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ of the Hamiltonian diagonal, rebuilt
        only when the interaction matrix changes (e.g. at the end of an SLM mask).
        """
        cached = self._interaction
        if (
            cached is None
            or cached[1].device != device
            or not torch.equal(cached[0], interaction_matrix)
        ):
            cached = (
                interaction_matrix.detach().clone(),
                make_interaction_diagonal(interaction_matrix, device),
            )
            self._interaction = cached
        return cached[1]

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
