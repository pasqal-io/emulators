import torch
from emu_base.lindblad_operators import compute_noise_from_lindbladians


dtype = torch.complex128
sigmax = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
sigmay = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype)
n_op = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)


class LindbladOperator:
    def __init__(
        self,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        pulser_linblads: list[torch.Tensor],
        interaction_matrix: torch.Tensor,
        device: torch.device,
    ):
        self.nqubits: int = len(omegas)
        self.omegas: torch.Tensor = omegas / 2.0
        self.deltas: torch.Tensor = deltas
        self.phis: torch.Tensor = phis
        self.interaction_matrix: torch.Tensor = interaction_matrix
        self.pulser_linblads: list[torch.Tensor] = pulser_linblads
        self.device: torch.device = device
        self.complex = self.phis.any()

        self.diag: torch.Tensor = self._create_diagonal()

    def _create_diagonal(self) -> torch.Tensor:
        """
        Return the diagonal elements of the Rydberg Hamiltonian matrix
        concerning only the interaction

            H.diag =  ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ
        """
        diag = torch.zeros(2**self.nqubits, dtype=dtype, device=self.device)

        for i in range(self.nqubits):
            diag = diag.view(2**i, 2, -1)
            i_fixed = diag[:, 1, :]
            for j in range(i + 1, self.nqubits):
                i_fixed = i_fixed.view(2**i, 2 ** (j - i - 1), 2, -1)
                # replacing i_j_fixed by i_fixed breaks the code :)
                i_j_fixed = i_fixed[:, :, 1, :]
                i_j_fixed += self.interaction_matrix[i, j]
        return diag.view(-1)

    def apply_local_op_to_density_matrix(
        self,
        density_matrix: torch.Tensor,
        local_op: torch.Tensor,
        target_qubit: int,
        op_conj_T: bool = False,
    ) -> torch.Tensor:
        """
        Apply a local operator A (2x2) matrix to a density matrix.
        If op_conj_T = True: L ρ L†
        Else: L ρ
        """

        orignal_shape = density_matrix.shape
        density_matrix = density_matrix.view(2**target_qubit, 2, -1)
        density_matrix = local_op @ density_matrix
        density_matrix = density_matrix.view(orignal_shape)

        if op_conj_T:
            density_matrix = density_matrix.view(
                2 ** (target_qubit + self.nqubits), 2, -1
            )
            density_matrix = local_op.conj() @ density_matrix
            density_matrix = density_matrix.view(orignal_shape)

        return density_matrix

    def __matmul__(self, density_matrix: torch.Tensor) -> torch.Tensor:
        """Apply the i*Lindblad operator :
         (Hρ - ρH) -0.5i ∑ₖ Lₖ† Lₖ ρ - ρ * 0.5i ∑ₖ Lₖ† Lₖ   + i*∑ₖ Lₖ ρ Lₖ†

        to the density matrix ρ
        """

        # compute -0.5i ∑ₖ Lₖ† Lₖ
        sum_lindblad_local = compute_noise_from_lindbladians(self.pulser_linblads).to(
            self.device
        )

        # apply local Hamiltonian terms (Ω σₓ - δ n - 0.5i ∑ₖ Lₖ† Lₖ) to each qubit
        H_local_den_matrix = torch.zeros_like(
            density_matrix, dtype=dtype, device=self.device
        )

        if not self.complex:
            for qubit, (omega, delta) in enumerate(zip(self.omegas, self.deltas)):
                H_q = (
                    omega * sigmax.to(device=self.device)
                    - delta * n_op.to(device=self.device)
                    + sum_lindblad_local
                )
                H_local_den_matrix += self.apply_local_op_to_density_matrix(
                    density_matrix, H_q, qubit
                )
        else:
            for qubit, (omega, delta, phi) in enumerate(
                zip(self.omegas, self.deltas, self.phis)
            ):
                H_q = (
                    omega
                    * (
                        (
                            torch.cos(phi) * sigmax.to(device=self.device)
                            + torch.sin(phi) * sigmay.to(device=self.device)
                        )
                    )
                    - delta * n_op.to(device=self.device)
                    + sum_lindblad_local
                )
                H_local_den_matrix += self.apply_local_op_to_density_matrix(
                    density_matrix, H_q, qubit
                )

        # apply diagonal interaction  ∑ᵢⱼ Uᵢⱼ nᵢ nⱼ
        diag_term = self.diag.view(-1, 1) * density_matrix  # elementwise column scaling
        H_den_matrix = H_local_den_matrix + diag_term

        # compute [H, ρ] - 0.5i ∑ₖ Lₖ† Lₖρ - ρ 0.5i ∑ₖ Lₖ† Lₖρ
        commutator = H_den_matrix - H_den_matrix.conj().T

        # compute ∑ₖ Lₖ ρ Lₖ† the last part of the Lindblad operator
        L_den_matrix_Ldag = sum(
            self.apply_local_op_to_density_matrix(
                density_matrix, L.to(self.device), qubit, op_conj_T=True
            )
            for qubit in range(self.nqubits)
            for L in self.pulser_linblads
        )

        return commutator + 1.0j * L_den_matrix_Ldag
