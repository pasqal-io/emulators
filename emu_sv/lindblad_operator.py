import torch

# from emu_sv.state_vector import StateVector


class LindbladOperator:
    def __init__(
        self,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        pulser_linblad: torch.Tensor,
        interaction_matrix: torch.Tensor,
        device: torch.device,
    ):
        self.nqubits: int = len(omegas)
        self.omegas: torch.Tensor = omegas / 2.0
        self.deltas: torch.Tensor = deltas
        self.phis: torch.Tensor = phis
        self.interaction_matrix: torch.Tensor = interaction_matrix
        self.device: torch.device = device

        self.diag: torch.Tensor = self._create_diagonal()

    def _create_diagonal(self) -> torch.Tensor:
        """
        Return the diagonal elements of the Rydberg Hamiltonian matrix

            H.diag = -∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ
        """
        diag = torch.zeros(2**self.nqubits, dtype=torch.complex128, device=self.device)

        for i in range(self.nqubits):
            diag = diag.view(2**i, 2, -1)
            i_fixed = diag[:, 1, :]
            i_fixed -= self.deltas[i]
            for j in range(i + 1, self.nqubits):
                i_fixed = i_fixed.view(2**i, 2 ** (j - i - 1), 2, -1)
                # replacing i_j_fixed by i_fixed breaks the code :)
                i_j_fixed = i_fixed[:, :, 1, :]
                i_j_fixed += self.interaction_matrix[i, j]
        return diag.view(-1)

    def __matmul__(self, densi_matrix: torch.Tensor) -> torch.Tensor:

        # (-∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ)|ψ❭
        pre_result = self.diag * densi_matrix.diag()
        # changing the diagonal of the densi_matrix
        densi_matrix[range(len(densi_matrix)), range(len(densi_matrix))] = pre_result

        # ∑ⱼΩⱼ/2[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ]|ψ❭
        # self._apply_sigma_operators(result, vec)
        return densi_matrix
