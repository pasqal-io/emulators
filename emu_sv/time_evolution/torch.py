import cmath
import math

import torch
from emu_base.math.double_krylov import double_krylov
from typing import Any, no_type_check
from emu_sv.hamiltonian import RydbergHamiltonian
from emu_base.math.krylov_exp import krylov_exp
from emu_base.math.chebyshev import chebyshev_exp

# The Chebyshev series is truncated where its error bound reaches this fraction of
# krylov_tolerance: the Lanczos stopping test overestimates the error (see
# docs/emu_sv/benchmarks/accuracy.md), whereas the Chebyshev bound is nearly attained.
CHEBYSHEV_TOLERANCE_FACTOR = 1e-2
# Steps with a larger alpha = r dt are split into equal sub-steps (rounding control).
CHEBYSHEV_MAX_ALPHA = 2000.0


def _apply_omega_real(
    result: torch.Tensor,
    i: int,
    inds: torch.Tensor,
    source: torch.Tensor,
    alpha: complex,
) -> None:
    """Accumulate to `result` the application of ασˣᵢ on `source`"""
    result.index_add_(i, inds, source, alpha=alpha)


def _apply_omega_complex(
    result: torch.Tensor,
    i: int,
    inds: torch.Tensor,
    source: torch.Tensor,
    alpha: complex,
) -> None:
    """Accumulate to `result` the application of ασ⁺ᵢ + α*σ⁻ᵢ on `source`"""
    result.index_add_(i, inds[0], source.select(i, 0).unsqueeze(i), alpha=alpha)
    result.index_add_(
        i,
        inds[1],
        source.select(i, 1).unsqueeze(2),
        alpha=alpha.conjugate(),
    )


class DHDOmegaSparse:
    """
    Derivative of the RydbergHamiltonian respect to Omega.
        ∂H/∂Ωₖ = 0.5[cos(ϕₖ)σˣₖ + sin(ϕₖ)σʸₖ]

    If ϕₖ=0, simplifies to ∂H/∂Ωₖ = 0.5σˣₖ
    """

    def __init__(self, index: int, device: torch.device, nqubits: int, phi: torch.Tensor):
        self.index = index
        self.shape = (2**index, 2, 2 ** (nqubits - index - 1))
        self.inds = torch.tensor([1, 0], device=device)  # flips the state, for 𝜎ₓ
        self.alpha = 0.5 * torch.exp(1j * phi).item()
        if phi.is_nonzero():
            self._apply_sigmas = _apply_omega_complex
        else:  # ∂H/∂Ωₖ = 0.5σˣₖ
            self._apply_sigmas = _apply_omega_real

    def __matmul__(self, vec: torch.Tensor) -> torch.Tensor:
        vec = vec.view(vec.shape[0], *self.shape)  # add batch dimension
        result = torch.zeros_like(vec)
        self._apply_sigmas(result, 2, self.inds, vec, alpha=self.alpha)
        return result.view(vec.shape[0], -1)


class DHDPhiSparse:
    """
    Derivative of the RydbergHamiltonian respect to Phi.
        ∂H/∂ϕₖ = 0.5Ωₖ[cos(ϕₖ+π/2)σˣₖ + sin(ϕₖ+π/2)σʸₖ]
    """

    def __init__(
        self,
        index: int,
        device: torch.device,
        nqubits: int,
        omega: torch.Tensor,
        phi: torch.Tensor,
    ):
        self.index = index
        self.shape = (2**index, 2, 2 ** (nqubits - index - 1))
        self.alpha = 0.5 * (omega * torch.exp(1j * (phi + torch.pi / 2))).item()
        self.inds = torch.tensor([1, 0], device=device)  # flips the state, for 𝜎ₓ

    def __matmul__(self, vec: torch.Tensor) -> torch.Tensor:
        vec = vec.view(vec.shape[0], *self.shape)  # add batch dimension
        result = torch.zeros_like(vec)
        _apply_omega_complex(result, 2, self.inds, vec, alpha=self.alpha)
        return result.view(vec.shape[0], -1)


class DHDDeltaSparse:
    """
    Derivative of the Rydberg Hamiltonian respect to Delta:
        ∂H/∂Δᵢ = -nᵢ
    """

    def __init__(self, i: int, nqubits: int):
        self.nqubits = nqubits
        self.shape = (2**i, 2, 2 ** (nqubits - i - 1))

    def __matmul__(self, vec: torch.Tensor) -> torch.Tensor:
        result = vec.clone()
        result = result.view(vec.shape[0], *self.shape)
        result[:, :, 0] = 0.0
        return -result.view(vec.shape[0], 2**self.nqubits)


class DHDUSparse:
    """
    Derivative of the Rydberg Hamiltonian respect to the interaction matrix:
        ∂H/∂Uᵢⱼ = nᵢnⱼ
    """

    def __init__(self, i: int, j: int, nqubits: int):
        self.shape = (2**i, 2, 2 ** (j - i - 1), 2, 2 ** (nqubits - j - 1))
        self.nqubits = nqubits

    def __matmul__(self, vec: torch.Tensor) -> torch.Tensor:
        result = vec.clone()
        result = result.view(vec.shape[0], *self.shape)
        result[:, :, 0] = 0.0
        result[:, :, 1, :, 0] = 0.0
        return result.view(vec.shape[0], 2**self.nqubits)


class DifferentiableStateVectorEvolution(torch.autograd.Function):
    """Custom autograd implementation of a step in the time evolution."""

    @staticmethod
    def get_hamiltonian(
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        interaction_matrix: torch.Tensor,
        device: torch.device,
        interaction_diag: torch.Tensor | None = None,
    ) -> RydbergHamiltonian:
        return RydbergHamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            interaction_matrix=interaction_matrix,
            device=device,
            noise=None,
            interaction_diag=interaction_diag,
        )

    @staticmethod
    def evolve(
        dt: float,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        interaction_matrix: torch.Tensor,
        state: torch.Tensor,
        krylov_tolerance: float,
        interaction_diag: torch.Tensor | None = None,
        use_chebyshev: bool = False,
    ) -> tuple[torch.Tensor, RydbergHamiltonian]:
        ham = DifferentiableStateVectorEvolution.get_hamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            interaction_matrix=interaction_matrix,
            device=state.device,
            interaction_diag=interaction_diag,
        )
        if use_chebyshev:
            d, lower, upper = ham.spectral_bounds()
            if not ham.omegas.any():  # H is diagonal
                return torch.exp(-1j * dt * d) * state, ham
            del d
            # exp(-i dt H) = exp(-i dt c) exp(-i α (H - c)/r), with c the shift, r the
            # radius and α = r dt: the spectrum of (H - c)/r lies in [-1, 1]
            shift, radius = (upper + lower) / 2, (upper - lower) / 2
            alpha = radius * dt
            substeps = max(1, math.ceil(abs(alpha) / CHEBYSHEV_MAX_ALPHA))
            shifted = DifferentiableStateVectorEvolution.get_hamiltonian(
                omegas=omegas,
                deltas=deltas,
                phis=phis,
                interaction_matrix=interaction_matrix,
                device=state.device,
                interaction_diag=ham.interaction_diag - shift,
            )
            op = shifted.make_op(2 / radius)
            phase = cmath.exp(-1j * dt * shift / substeps)
            tolerance = krylov_tolerance * CHEBYSHEV_TOLERANCE_FACTOR / substeps
            res = state
            for _ in range(substeps):
                res = chebyshev_exp(op, res, alpha / substeps, phase, tolerance)
            return res, ham
        res = krylov_exp(
            ham.make_op(-1j * dt),
            state,
            norm_tolerance=krylov_tolerance,
            exp_tolerance=krylov_tolerance,
            is_hermitian=True,
        )
        return res, ham

    @torch.autograd.function.once_differentiable
    @staticmethod
    def forward(
        ctx: Any,
        dt: float,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        interaction_matrix: torch.Tensor,
        state: torch.Tensor,
        krylov_tolerance: float,
        interaction_diag: torch.Tensor | None = None,
        use_chebyshev: bool = False,
    ) -> tuple[torch.Tensor, RydbergHamiltonian]:
        """
        Returns the time evolved state
            |ψ(t+dt)〉= exp(-i dt H)|ψ(t)〉
        under the Hamiltonian H built from the input Tensor parameters, omegas, deltas, phis and
        the interaction matrix.

        Args:
            ctx (Any): context object to stash information for backward computation.
            dt (float): timestep
            omegas (torch.Tensor): 1D tensor of driving strengths for each qubit.
            deltas (torch.Tensor): 1D tensor of detuning values for each qubit.
            phis (torch.Tensor): 1D tensor of phase values for each qubit.
            interaction_matrix (torch.Tensor): matrix representing the interaction
                strengths between each pair of qubits.
            state (Tensor): input state to be evolved
            krylov_tolerance (float): tolerance for krylov_exp
            interaction_diag (torch.Tensor, optional): the interaction part of the
                Hamiltonian diagonal, if already computed (see `RydbergHamiltonian`).
            use_chebyshev (bool): expand exp(-i dt H) in Chebyshev polynomials instead
                of using the Lanczos algorithm (forward step only).
        """
        res, ham = DifferentiableStateVectorEvolution.evolve(
            dt,
            omegas,
            deltas,
            phis,
            interaction_matrix,
            state,
            krylov_tolerance,
            interaction_diag,
            use_chebyshev,
        )
        ctx.save_for_backward(
            omegas, deltas, phis, interaction_matrix, state, interaction_diag
        )
        ctx.dt = dt
        ctx.tolerance = krylov_tolerance
        return res, ham

    # mypy complains and I don't know why
    # backward expects same number of gradients as output of forward, gham is unused
    @no_type_check
    @staticmethod
    def backward(ctx: Any, grad_state_out: torch.Tensor, gham: None) -> tuple[
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
    ]:
        """
        In the backward pass we receive a Tensor containing the gradient of the loss L
        with respect to the output
            |gψ(t+dt)〉= ∂L/∂|ψ(t+dt)〉,
        and return the gradients of the loss with respect to the input tensors parameters
            - gΩⱼ = ∂L/∂Ωⱼ =〈gψ(t+dt)|dU(H,∂H/∂Ωⱼ)|ψ(t)〉
            - gΔⱼ = ∂L/∂Δⱼ =  ...
            - |gψ(t)〉= ∂L/∂|ψ(t)〉= exp(i dt H)|gψ(t+dt)〉

        Args:
            ctx (Any): context object to stash information for backward computation.
            grad_state_out (torch.Tensor): |gψ(t+dt)〉

        Return:
            grad_omegas (torch.Tensor): 1D tensor of gradients with respect to Ωⱼ for each qubit.
            grad_deltas (torch.Tensor): 1D tensor of gradients with respect to Δⱼ for each qubit.
            grad_phis (torch.Tensor): 1D tensor of gradients with respect to φⱼ for each qubit.
            grad_state_in (torch.Tensor): 1D tensor gradient with respect to the input state.

        Notes:
        Gradients are obtained by matching the total variations
            〈gψ(t+dt)|d|ψ(t+dt)〉= ∑ⱼgΔⱼ*dΔⱼ + ∑ⱼgΩⱼ*dΩⱼ + ∑ⱼgφ*dφⱼ +〈gψ(t)|d|ψ(t)〉  (1)

        For the exponential map U = exp(-i dt H), differentiating reads:
            d|ψ(t+dt)〉= dU|ψ(t)〉+ Ud|ψ(t)〉
            dU = ∑ⱼdU(H,∂H/∂Δⱼ) + ∑ⱼdU(H,∂H/∂Ωⱼ) + ∑ⱼdU(H,∂H/∂φⱼ)                      (2)

        where dU(H,E) is the Fréchet derivative of the exponential map
        along the direction E:
        - https://eprints.maths.manchester.ac.uk/1218/1/covered/MIMS_ep2008_26.pdf
        - https://en.wikipedia.org/wiki/Derivative_of_the_exponential_map

        Substituting (2) into (1) leads to the expressions of the gradients
        with respect to the input tensors above.

        Variations with respect to the Hamiltonian parameters are computed as
            gΩ = 〈gψ(t+dt)|dU(H,∂H/∂Ω)|ψ(t)〉
               = Tr( -i dt ∂H/∂Ω @ dU(H,|ψ(t)〉〈gψ(t+dt)|) ),
        where under the trace sign, ∂H/∂Ω and |ψ(t)〉〈gψ(t+dt)| can be switched.

        - The Fréchet derivative is computed in a Arnoldi-Gram-Schmidt
        decomposition in the `double_krylov` method:
            dU(H,|a〉〈b|) = Va @ dS @ Vb*
        where Va,Vb are orthogonal Krylov basis associated
        with |a〉and |b〉respectively.

        - The action of the derivatives of the Hamiltonian with
        respect to the input parameters are implemented separately in
            - ∂H/∂Ω:  `DHDOmegaSparse`
            - ∂H/∂Δ:  `DHDDeltaSparse`
            - ∂H/∂φ:  `DHDPhiSparse`
            - ∂H/∂Uᵢⱼ `DHDUSparse`

        Then, the resulting gradient respect to a generic parameter reads:
            gΩ = Tr( -i dt ∂H/∂Ω @ Vs @ dS @ Vg* )
        """
        omegas, deltas, phis, interaction_matrix, state, interaction_diag = (
            ctx.saved_tensors
        )
        dt = ctx.dt
        tolerance = ctx.tolerance
        nqubits = len(omegas)

        grad_omegas, grad_deltas, grad_phis = None, None, None
        grad_int_mat = None
        grad_state_in = None

        ham = DifferentiableStateVectorEvolution.get_hamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            interaction_matrix=interaction_matrix,
            device=state.device,
            interaction_diag=interaction_diag,
        )

        if any(ctx.needs_input_grad[1:5]):
            lanczos_vectors_state, dS, lanczos_vectors_grad = double_krylov(
                ham.make_op(-1j * dt), state, grad_state_out, tolerance
            )
            # TODO: explore returning directly the basis in matrix form
            Vs = torch.stack(lanczos_vectors_state)
            del lanczos_vectors_state
            Vg = torch.stack(lanczos_vectors_grad)
            del lanczos_vectors_grad
            e_l = dS.mT @ Vs

        if ctx.needs_input_grad[1]:
            grad_omegas = torch.zeros_like(omegas)
            for i in range(nqubits):
                # dh as per the docstring
                dho = DHDOmegaSparse(i, e_l.device, nqubits, phis[i])
                # compute the trace
                v = dho @ e_l
                grad_omegas[i] = (-1j * dt * torch.tensordot(Vg.conj(), v)).real

        if ctx.needs_input_grad[2]:
            grad_deltas = torch.zeros_like(deltas)
            for i in range(nqubits):
                dhd = DHDDeltaSparse(i, nqubits)
                v = dhd @ e_l
                grad_deltas[i] = (-1j * dt * torch.tensordot(Vg.conj(), v)).real

        if ctx.needs_input_grad[3]:
            grad_phis = torch.zeros_like(phis)
            for i in range(nqubits):
                dhp = DHDPhiSparse(i, e_l.device, nqubits, omegas[i], phis[i])
                v = dhp @ e_l
                grad_phis[i] = (-1j * dt * torch.tensordot(Vg.conj(), v)).real

        if ctx.needs_input_grad[4]:
            grad_int_mat = torch.zeros_like(interaction_matrix)
            for i in range(nqubits):
                for j in range(i + 1, nqubits):
                    dhu = DHDUSparse(i, j, nqubits)
                    v = dhu @ e_l
                    grad_int_mat[i, j] = (-1j * dt * torch.tensordot(Vg.conj(), v)).real

        if ctx.needs_input_grad[5]:
            grad_state_in = krylov_exp(
                ham.make_op(1j * dt), grad_state_out.detach(), tolerance, tolerance
            )

        return (
            None,
            grad_omegas,
            grad_deltas,
            grad_phis,
            grad_int_mat,
            grad_state_in,
            None,
            None,
            None,
            None,
        )
