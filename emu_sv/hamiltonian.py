from typing import Callable

import torch
from emu_sv.state_vector import StateVector
from emu_sv import hamiltonian_kernels as kernels


def make_interaction_diagonal(
    interaction_matrix: torch.Tensor, device: torch.device | str
) -> torch.Tensor:
    """
    Return the interaction part ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ of the Hamiltonian diagonal, as a real
    vector of length 2**N. Building it takes N(N-1)/2 passes over that vector, while
    the interaction matrix rarely changes during a sequence: the time steppers keep
    it between time steps (see `BaseStepper.interaction_diagonal`).
    """
    nqubits = interaction_matrix.shape[0]
    diag = torch.zeros(2**nqubits, dtype=torch.float64, device=device)
    for i in range(nqubits):
        diag = diag.view(2**i, 2, -1)
        i_fixed = diag[:, 1, :]
        for j in range(i + 1, nqubits):
            i_fixed = i_fixed.view(2**i, 2 ** (j - i - 1), 2, -1)
            # replacing i_j_fixed by i_fixed breaks the code :)
            i_j_fixed = i_fixed[:, :, 1, :]
            i_j_fixed += interaction_matrix[i, j]
    return diag.view(-1)


class RydbergHamiltonian:
    """Represents the Rydberg Hamiltonian for a system of interacting qubits
    driven by laser fields, including detuning, phase, and interaction terms.

    The Hamiltonian is defined as:

        H = ∑ⱼ (Ωⱼ/2)[cos(ϕⱼ) σˣⱼ + sin(ϕⱼ) σʸⱼ] - ∑ⱼ Δⱼ nⱼ + ∑_{i>j} Uᵢⱼ nᵢ nⱼ

    where:
        - Ωⱼ is the Rabi frequency on qubit j,
        - Δⱼ is the detuning on qubit j,
        - ϕⱼ is the laser phase on qubit j,
        - Uᵢⱼ is the interaction strength between qubits i and j,
        - nⱼ = |1⟩⟨1| is the number operator on qubit j.

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
        diag (torch.Tensor): diagonal contribution to the Hamiltonian (detuning + interactions),
            built on first use.
        interaction_diag (torch.Tensor): real interaction part ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ of the diagonal.
        inds (torch.Tensor): index mapping for σˣ operations.
        nqubits (int): number of qubits in the system.

    Methods:
        __mul__(vec): Applies the Hamiltonian H to a state vector |ψ⟩.
        _apply_sigma_operators_real(): Applies only σˣ terms (ϕⱼ = 0).
        _apply_sigma_operators_complex(): Applies generalized σˣ/σʸ terms (ϕⱼ ≠ 0).
        _create_diagonal(): Computes the diagonal part of H from Δⱼ and Uᵢⱼ.
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
        interaction_diag: torch.Tensor | None = None,
    ):
        self.nqubits: int = len(omegas)
        self.omegas: torch.Tensor = omegas / 2.0
        self.deltas: torch.Tensor = deltas
        self.phis: torch.Tensor = phis
        self.interaction_matrix: torch.Tensor = interaction_matrix
        self.device: torch.device = device
        self.noise: torch.Tensor | None = noise

        if interaction_diag is None:
            interaction_diag = make_interaction_diagonal(interaction_matrix, device)
        self.interaction_diag: torch.Tensor = interaction_diag
        self._diag: torch.Tensor | None = None
        self._kernel_buffer: torch.Tensor | None = None
        # the fused kernel does not record operations for autograd
        self._requires_grad = any(
            t.requires_grad for t in (omegas, deltas, phis, interaction_matrix)
        )
        self.inds = torch.tensor([1, 0], device=self.device)  # flips the state, for σˣ
        self.complex = bool(self.phis.any()) or noise is not None
        if self.complex:
            self.omegas_c = self.omegas * torch.exp(1.0j * self.phis)
            self.omegas_c = torch.stack((self.omegas_c, self.omegas_c.conj()), dim=1)
            if noise is not None:
                self.omegas_c[:, 0] += noise[1, 0]
                self.omegas_c[:, 1] += noise[0, 1]

    @property
    def diag(self) -> torch.Tensor:
        """Diagonal of H (complex128), built on first use."""
        if self._diag is None:
            self._diag = self._create_diagonal()
        return self._diag

    def _use_kernel(self, vec: torch.Tensor) -> bool:
        return kernels.is_available(vec) and not (
            torch.is_grad_enabled() and (vec.requires_grad or self._requires_grad)
        )

    def _kernel_args(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Per-qubit coefficients of the fused kernel (see `hamiltonian_kernels`) and the
        scalar 1, copied to the device once per Hamiltonian.
        """
        if self._kernel_buffer is None:
            d0 = torch.zeros(self.nqubits, dtype=torch.complex128)
            d1 = -self.deltas.detach().cpu().to(torch.complex128)
            if self.complex:
                omegas_c = self.omegas_c.detach().cpu()
                f1, f0 = omegas_c[:, 0], omegas_c[:, 1]
            else:
                f0 = f1 = self.omegas.detach().cpu().to(torch.complex128)
            if self.noise is not None:
                noise = self.noise.detach().cpu()
                d0 += noise[0, 0]
                d1 += noise[1, 1]
            one = torch.ones(1, dtype=torch.complex128)
            buffer = torch.cat((torch.stack((d0, d1, f0, f1), dim=1).view(-1), one))
            self._kernel_buffer = buffer.to(self.interaction_diag.device)
        table = torch.view_as_real(self._kernel_buffer[:-1]).reshape(-1)
        return table, self._kernel_buffer[-1:]

    def make_op(self, factor: complex) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Return the operator x ↦ factor·(H x), e.g. -i dt H for `krylov_exp`.
        With the fused kernel, the factor is applied inside the kernel.
        """
        factor_on_device: torch.Tensor | None = None

        def op(x: torch.Tensor) -> torch.Tensor:
            nonlocal factor_on_device
            if not self._use_kernel(x):
                return factor * (self * x)
            if factor_on_device is None:
                factor_on_device = torch.tensor(
                    [factor], dtype=torch.complex128, device=x.device
                )
            table, _ = self._kernel_args()
            return kernels.apply(
                x, self.interaction_diag, table, factor_on_device, self.complex
            )

        return op

    def __mul__(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Apply the `RydbergHamiltonian` to the input state vector, i.e. H*|ψ❭.

        - The diagonal part of the Hamiltonian (Δⱼ and Uᵢⱼ terms) is stored and
        applyed directly as H.diag*|ψ❭.
        - The off-diagonal part (Ωⱼ and ϕⱼ terms) are applied sequentially across
        qubit indices in `self._apply_sigma_operators`.

        Args:
            vec (torch.Tensor): vec (torch.Tensor): the input state vector.

        On a CUDA device with Triton available, both parts are applied in a single
        pass by the fused kernel of `emu_sv.hamiltonian_kernels`.

        Returns:
            the resulting state vector.
        """
        if self._use_kernel(vec):
            table, one = self._kernel_args()
            return kernels.apply(vec, self.interaction_diag, table, one, self.complex)
        # (-∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ)|ψ❭
        result = self.diag * vec
        # ∑ⱼΩⱼ/2[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ]|ψ❭
        if self.complex:
            self._apply_sigma_operators_complex(result, vec)
        else:
            self._apply_sigma_operators_real(result, vec)
        return result

    def _apply_sigma_operators_real(
        self, result: torch.Tensor, vec: torch.Tensor
    ) -> None:
        """
        Apply the ∑ⱼ(Ωⱼ/2)σˣⱼ operator to the input vector |ψ❭.

        Args:
            vec (torch.Tensor): the input state vector.

        Returns:
            the resulting state vector.
        """
        dim_to_act = 1
        for n, omega_n in enumerate(self.omegas):
            shape_n = (2**n, 2, 2 ** (self.nqubits - n - 1))
            vec = vec.view(shape_n)
            result = result.view(shape_n)
            result.index_add_(dim_to_act, self.inds, vec, alpha=omega_n)

    def _apply_sigma_operators_complex(
        self, result: torch.Tensor, vec: torch.Tensor
    ) -> None:
        """
        Apply the ∑ⱼΩⱼ/2[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ] operator to the input vector |ψ❭.

        Args:
            vec (torch.Tensor): the input state vector.

        Returns:
            the resulting state vector.
        """
        dim_to_act = 1
        for n, c_omega_n in enumerate(self.omegas_c):
            shape_n = (2**n, 2, 2 ** (self.nqubits - n - 1))
            vec = vec.view(shape_n)
            result = result.view(shape_n)
            result.index_add_(
                dim_to_act, self.inds[0], vec[:, 0, :].unsqueeze(1), alpha=c_omega_n[0]
            )
            result.index_add_(
                dim_to_act,
                self.inds[1],
                vec[:, 1, :].unsqueeze(1),
                alpha=c_omega_n[1],
            )

    def _create_diagonal(self) -> torch.Tensor:
        """
        Return the diagonal elements of the Rydberg Hamiltonian matrix

            H.diag = -∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ

        from the real interaction part `interaction_diag`.
        """
        diag = self.interaction_diag.to(torch.complex128)

        if self.noise is not None:
            for i in range(self.nqubits):
                diag = diag.view(2**i, 2, -1)
                diag[:, 0, :] += self.noise[0, 0]
                diag[:, 1, :] += self.noise[1, 1]

        for i in range(self.nqubits):
            diag = diag.view(2**i, 2, -1)
            i_fixed = diag[:, 1, :]
            i_fixed -= self.deltas[i]
        return diag.view(-1)

    def expect(self, state: StateVector) -> torch.Tensor:
        """Return the energy expectation value E=❬ψ|H|ψ❭"""
        assert isinstance(
            state, StateVector
        ), "Currently, only expectation values of StateVectors are supported"
        en = torch.vdot(state.data, self * state.data)
        if self.noise is None:
            assert torch.allclose(en.imag, torch.zeros_like(en.imag), atol=1e-8)
        return en.real  # if there is lindblad noise, there is non-zero imaginary part.
