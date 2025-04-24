import torch
from emu_base.lindblad_operators import compute_noise_from_lindbladians


dtype = torch.complex128
sigmax = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
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

        self.diag: torch.Tensor = self._create_diagonal()

    def _create_diagonal(self) -> torch.Tensor:
        """
        Return the diagonal elements of the Rydberg Hamiltonian matrix
        concerning the interaction

            H.diag =  ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ
        """
        diag = torch.zeros(2**self.nqubits, dtype=dtype, device=self.device)

        for i in range(self.nqubits):
            diag = diag.view(2**i, 2, -1)
            i_fixed = diag[:, 1, :]
            # i_fixed -= self.deltas[i]
            for j in range(i + 1, self.nqubits):
                i_fixed = i_fixed.view(2**i, 2 ** (j - i - 1), 2, -1)
                # replacing i_j_fixed by i_fixed breaks the code :)
                i_j_fixed = i_fixed[:, :, 1, :]
                i_j_fixed += self.interaction_matrix[i, j]
        return diag.view(-1)

    def apply_local_operator_to_density_matrix_to_local_op(
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
        nqubits = self.nqubits
        dim = 2 ** (nqubits - 1)

        # reshape density matrix to 2n-dimensional tensor
        rho = density_matrix.view([2] * 2 * nqubits)

        # permute to bring target_qubit to front in both bra and ket indices
        perm = list(range(nqubits))
        perm.remove(target_qubit)
        bra_perm = [target_qubit] + perm
        ket_perm = [x + nqubits for x in bra_perm]
        rho = rho.permute(bra_perm + ket_perm)

        # reshape and contiguous is needed here
        rho = rho.contiguous().view(2, -1)  # (2, dim*2*dim)

        # apply local_op to bra (left multiply)
        # A (2x2) @ rho (2, -1) -> (2, dim*2*dim)
        rho = local_op @ rho
        rho = rho.view(2, dim, 2, dim)

        # if op_conj_T apply local_op† to ket (right multiply)
        if op_conj_T:
            # transpose (complex conjugate) and move 3rd axis to front for multiplication
            rho = rho.permute(0, 1, 3, 2).contiguous()  # (2, dim, dim, 2)
            rho = rho.view(-1, 2) @ local_op.conj().T  # (2*dim*dim, 2)
            rho = rho.view(2, dim, dim, 2).permute(0, 1, 3, 2)  # (2, dim, 2, dim)

        # reshape back
        rho = rho.view([2] * 2 * nqubits)
        inv_perm = [0] * (2 * nqubits)
        for i, p in enumerate(bra_perm + ket_perm):
            inv_perm[p] = i
        rho = rho.permute(inv_perm)

        return rho.contiguous().view(2**nqubits, 2**nqubits)

    def __matmul__(self, density_matrix: torch.Tensor) -> torch.Tensor:
        """Apply the i*Lindblad operator :
         (Hρ - ρH) -0.5i ∑ₖ Lₖ† Lₖ ρ - ρ * 0.5i ∑ₖ Lₖ† Lₖ   + i*∑ₖ Lₖ ρ Lₖ†

        to the density matrix ρ
        """

        # compute -0.5i ∑ₖ Lₖ† Lₖ (taken from the Lindblad class)
        sum_lindblad_local = compute_noise_from_lindbladians(self.pulser_linblads)

        # apply local Hamiltonian terms (Ω σₓ - δ n - 0.5i ∑ₖ Lₖ† Lₖ) to each qubit
        H_local_rho = torch.zeros_like(density_matrix, dtype=dtype)
        for qubit, (omega, delta) in enumerate(zip(self.omegas, self.deltas)):
            H_q = omega * sigmax - delta * n_op + sum_lindblad_local
            H_local_rho += self.apply_local_operator_to_density_matrix_to_local_op(
                density_matrix, H_q, qubit
            )

        # apply diagonal interaction  ∑ᵢⱼ Uᵢⱼ nᵢ nⱼ
        diag_term = self.diag.view(-1, 1) * density_matrix  # elementwise column scaling
        H_rho = H_local_rho + diag_term

        # compute [H, ρ] - 0.5i ∑ₖ Lₖ† Lₖρ - ρ 0.5i ∑ₖ Lₖ† Lₖρ
        commutator = H_rho - H_rho.conj().T

        # compute ∑ₖ Lₖ ρ Lₖ† the last part of the Lindblad operator
        L_rho_Ldag = sum(
            self.apply_local_operator_to_density_matrix_to_local_op(
                density_matrix, L, qubit, op_conj_T=True
            )
            for qubit in range(self.nqubits)
            for L in self.pulser_linblads
        )

        return commutator + 1.0j * L_rho_Ldag
