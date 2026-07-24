import torch
from emu_sv.state_vector import StateVector


def _fwht(vec: torch.Tensor, nqubits: int) -> torch.Tensor:
    """
    Unnormalized fast Walsh-Hadamard transform √2·Hd applied to every qubit.

    Applied twice, it equals 2ᴺ·𝟙; the missing 2⁻ᴺ is folded into the
    diagonals of `XYHamiltonian`. The input is modified in-place.
    """
    for n in range(nqubits):
        x = vec.view(2**n, 2, -1)
        x[:, 0] += x[:, 1]  # a+b
        x[:, 1] = x[:, 0] - 2.0 * x[:, 1]  # a-b
    return vec.view(-1)


def _ywht_dag(vec: torch.Tensor, nqubits: int) -> torch.Tensor:
    """
    Unnormalized √2·V† applied to every qubit, where V = S·Hd diagonalizes
    σʸ = V σᶻ V†. The input is modified in-place.
    """
    for n in range(nqubits):
        x = vec.view(2**n, 2, -1)
        x[:, 0] -= 1.0j * x[:, 1]  # a-ib
        x[:, 1] = x[:, 0] + 2.0j * x[:, 1]  # a+ib
    return vec.view(-1)


def _ywht(vec: torch.Tensor, nqubits: int) -> torch.Tensor:
    """
    Unnormalized √2·V applied to every qubit (inverse of `_ywht_dag` up to
    the factor 2ᴺ folded into the diagonals). The input is modified in-place.
    """
    for n in range(nqubits):
        x = vec.view(2**n, 2, -1)
        x[:, 0] += x[:, 1]  # a+b
        x[:, 1] = 1.0j * (x[:, 0] - 2.0 * x[:, 1])  # i(a-b)
    return vec.view(-1)


class XYHamiltonian:
    """Represents the XY Hamiltonian for a system of interacting qubits
    driven by laser fields, including detuning, phase, and interaction terms.

    The Hamiltonian is defined as:

        H = ∑ⱼ (Ωⱼ/2)[cos(ϕⱼ) σˣⱼ + sin(ϕⱼ) σʸⱼ] - ∑ⱼ Δⱼ nⱼ
            + ∑_{i>j} Uᵢⱼ (σˣᵢσˣⱼ + σʸᵢσʸⱼ)

    where:
        - Ωⱼ is the Rabi frequency on qubit j,
        - Δⱼ is the detuning on qubit j,
        - ϕⱼ is the laser phase on qubit j,
        - Uᵢⱼ is the interaction strength between qubits i and j,
        - nⱼ = |1⟩⟨1| is the number operator on qubit j.

    Note that σˣᵢσˣⱼ + σʸᵢσʸⱼ = 2(σ⁺ᵢσ⁻ⱼ + σ⁻ᵢσ⁺ⱼ), i.e. the interaction
    is twice the Pulser XY convention.

    Unlike the diagonal nᵢnⱼ interaction of the `RydbergHamiltonian`, the
    XY interaction is off-diagonal. Applying it pair by pair would cost
    O(N²·2ᴺ) per application. Instead, each part is conjugated into a basis
    where it is diagonal:

        σˣ = Hd σᶻ Hd    and    σʸ = V σᶻ V†  with  V = S·Hd,

    so that

        H|ψ❭ = z_diag⊙|ψ❭ + Hd⊗ᴺ(x_diag ⊙ Hd⊗ᴺ|ψ❭) + V⊗ᴺ(y_diag ⊙ V†⊗ᴺ|ψ❭)

    where the σˣ (resp. σʸ) drive terms fold into x_diag (resp. y_diag)
    alongside the shared interaction diagonal ∑_{i>j}Uᵢⱼ(-1)^(sᵢ+sⱼ). Each
    transform costs O(N·2ᴺ), so a full application costs O(N·2ᴺ) instead of
    O(N²·2ᴺ).

    Attributes:
        omegas (torch.Tensor): vector of Rabi frequencies Ωⱼ / 2 for each qubit.
        deltas (torch.Tensor): vector of detunings Δⱼ for each qubit.
        phis (torch.Tensor): vector of phases ϕⱼ for each qubit.
        interaction_matrix (torch.Tensor): matrix Uᵢⱼ for pairwise interactions.
        device (torch.device): device on which all tensors are allocated.
        noise (torch.Tensor, optional): The single-qubit noise
            term -0.5i∑ⱼLⱼ†Lⱼ applied to all qubits.
            This can be computed using the `compute_noise_from_lindbladians`
            function.
        z_diag (torch.Tensor): diagonal contribution in the computational basis
            (detuning + diagonal noise).
        x_diag (torch.Tensor): diagonal contribution in the Hadamard basis
            (interaction + σˣ drive), scaled by 2⁻ᴺ.
        y_diag (torch.Tensor): diagonal contribution in the σʸ eigenbasis
            (interaction + σʸ drive), scaled by 2⁻ᴺ.
        nqubits (int): number of qubits in the system.

    Methods:
        __mul__(vec): Applies the Hamiltonian H to a state vector |ψ⟩.
        expect(state): Computes ⟨ψ|H|ψ⟩ for a given StateVector.
    """

    def __init__(
        self,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        interaction_matrix: torch.Tensor,
        device: torch.device,
        noise: torch.Tensor | None,
    ):
        self.nqubits: int = len(omegas)
        self.omegas: torch.Tensor = omegas / 2.0
        self.deltas: torch.Tensor = deltas
        self.phis: torch.Tensor = phis
        self.interaction_matrix: torch.Tensor = interaction_matrix
        self.device: torch.device = device
        self.noise: torch.Tensor | None = noise

        self.z_diag: torch.Tensor = self._create_z_diagonal()
        self.x_diag, self.y_diag = self._create_xy_diagonals()

    def __mul__(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Apply the `XYHamiltonian` to the input state vector, i.e. H*|ψ❭.

        - The detuning terms (Δⱼ) are diagonal and applied directly as
        H.z_diag*|ψ❭.
        - The interaction (Uᵢⱼ) and drive (Ωⱼ, ϕⱼ) terms are applied as
        diagonals in the σˣ and σʸ eigenbases, conjugated by the
        corresponding basis transforms.

        Args:
            vec (torch.Tensor): the input state vector.

        Returns:
            the resulting state vector.
        """
        # (-∑ⱼΔⱼnⱼ)|ψ❭
        result = self.z_diag * vec
        # (∑ⱼΩⱼ/2 cos(ϕⱼ)σˣⱼ + ∑ᵢ﹥ⱼUᵢⱼσˣᵢσˣⱼ)|ψ❭
        u = _fwht(vec.clone(), self.nqubits)
        u *= self.x_diag
        result += _fwht(u, self.nqubits)
        # (∑ⱼΩⱼ/2 sin(ϕⱼ)σʸⱼ + ∑ᵢ﹥ⱼUᵢⱼσʸᵢσʸⱼ)|ψ❭
        u = _ywht_dag(vec.clone(), self.nqubits)
        u *= self.y_diag
        result += _ywht(u, self.nqubits)
        return result

    def _create_z_diagonal(self) -> torch.Tensor:
        """
        Return the diagonal of the XY Hamiltonian in the computational basis

            H.z_diag = -∑ⱼΔⱼnⱼ

        plus the diagonal part of the noise, if any.
        """
        diag = torch.zeros(2**self.nqubits, dtype=torch.complex128, device=self.device)

        for i in range(self.nqubits):
            diag = diag.view(2**i, 2, -1)
            diag[:, 1, :] -= self.deltas[i]
            if self.noise is not None:
                diag[:, 0, :] += self.noise[0, 0]
                diag[:, 1, :] += self.noise[1, 1]
        return diag.view(-1)

    def _create_xy_diagonals(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the diagonals of the σˣ and σʸ parts of the XY Hamiltonian in
        their respective eigenbases,

            H.x_diag = 2⁻ᴺ[∑ⱼΩⱼ/2 cos(ϕⱼ)σᶻⱼ + ∑ᵢ﹥ⱼUᵢⱼσᶻᵢσᶻⱼ]
            H.y_diag = 2⁻ᴺ[∑ⱼΩⱼ/2 sin(ϕⱼ)σᶻⱼ + ∑ᵢ﹥ⱼUᵢⱼσᶻᵢσᶻⱼ]

        plus the off-diagonal part of the noise, if any. The factor 2⁻ᴺ
        compensates the unnormalized basis transforms.
        """
        # ∑ᵢ﹥ⱼUᵢⱼσᶻᵢσᶻⱼ is shared between both diagonals
        diag = torch.zeros(2**self.nqubits, dtype=torch.complex128, device=self.device)
        for i in range(self.nqubits):
            for j in range(i + 1, self.nqubits):
                u_ij = self.interaction_matrix[i, j]
                pair = diag.view(2**i, 2, 2 ** (j - i - 1), 2, -1)
                pair[:, 0, :, 0, :] += u_ij
                pair[:, 1, :, 1, :] += u_ij
                pair[:, 0, :, 1, :] -= u_ij
                pair[:, 1, :, 0, :] -= u_ij

        x_diag = diag.clone()
        y_diag = diag
        # the σˣ (σʸ) drive and noise terms are diagonal in the same basis
        # as σˣᵢσˣⱼ (σʸᵢσʸⱼ): fold them into the respective diagonals
        x_coeffs = self.omegas * torch.cos(self.phis) + 0.0j
        y_coeffs = self.omegas * torch.sin(self.phis) + 0.0j
        if self.noise is not None:
            # n₀₁|0❭❬1| + n₁₀|1❭❬0| = (n₀₁+n₁₀)/2 σˣ + i(n₀₁-n₁₀)/2 σʸ
            x_coeffs += (self.noise[0, 1] + self.noise[1, 0]) / 2.0
            y_coeffs += 1.0j * (self.noise[0, 1] - self.noise[1, 0]) / 2.0
        for i in range(self.nqubits):
            x_view = x_diag.view(2**i, 2, -1)
            x_view[:, 0, :] += x_coeffs[i]
            x_view[:, 1, :] -= x_coeffs[i]
            y_view = y_diag.view(2**i, 2, -1)
            y_view[:, 0, :] += y_coeffs[i]
            y_view[:, 1, :] -= y_coeffs[i]

        norm = 2.0 ** (-self.nqubits)
        return x_diag * norm, y_diag * norm

    def expect(self, state: StateVector) -> torch.Tensor:
        """Return the energy expectation value E=❬ψ|H|ψ❭"""
        assert isinstance(
            state, StateVector
        ), "Currently, only expectation values of StateVectors are supported"
        en = torch.vdot(state.data, self * state.data)
        return en.real  # if there is lindblad noise, there is non-zero imaginary part.
