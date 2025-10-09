import torch
from typing import Any, no_type_check
from emu_base.math.krylov_exp import krylov_exp
from emu_base.math.double_krylov import double_krylov
from emu_sv.hamiltonian import RydbergHamiltonian
from emu_sv.lindblad_operator import RydbergLindbladian


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


class EvolveStateVector(torch.autograd.Function):
    """Custom autograd implementation of a step in the time evolution."""

    @staticmethod
    def evolve(
        dt: float,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        interaction_matrix: torch.Tensor,
        state: torch.Tensor,
        krylov_tolerance: float,
    ) -> tuple[torch.Tensor, RydbergHamiltonian]:
        ham = RydbergHamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            interaction_matrix=interaction_matrix,
            device=state.device,
        )

        def op(x: torch.Tensor) -> torch.Tensor:
            return -1j * dt * (ham * x)

        res = krylov_exp(
            op,
            state,
            norm_tolerance=krylov_tolerance,
            exp_tolerance=krylov_tolerance,
            is_hermitian=True,
        )
        return res, ham

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
            krylov_tolerance (float):
        """
        res, ham = EvolveStateVector.evolve(
            dt, omegas, deltas, phis, interaction_matrix, state, krylov_tolerance
        )
        ctx.save_for_backward(omegas, deltas, phis, interaction_matrix, state)
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
        omegas, deltas, phis, interaction_matrix, state = ctx.saved_tensors
        dt = ctx.dt
        tolerance = ctx.tolerance
        nqubits = len(omegas)

        grad_omegas, grad_deltas, grad_phis = None, None, None
        grad_int_mat = None
        grad_state_in = None

        ham = RydbergHamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            interaction_matrix=interaction_matrix,
            device=state.device,
        )

        if any(ctx.needs_input_grad[1:5]):

            def op(x: torch.Tensor):
                return -1j * dt * (ham * x)

            lanczos_vectors_state, dS, lanczos_vectors_grad = double_krylov(
                op, state, grad_state_out, tolerance
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

            def op(x: torch.Tensor):
                return (1j * dt) * (ham * x)

            grad_state_in = krylov_exp(op, grad_state_out.detach(), tolerance, tolerance)

        return (
            None,
            grad_omegas,
            grad_deltas,
            grad_phis,
            grad_int_mat,
            grad_state_in,
            None,
        )


class EvolveDensityMatrix:
    """Evolution of a density matrix under a Lindbladian operator."""

    @staticmethod
    def evolve(
        dt: float,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        full_interaction_matrix: torch.Tensor,
        density_matrix: torch.Tensor,
        krylov_tolerance: float,
        pulser_lindblads: list[torch.Tensor],
    ) -> tuple[torch.Tensor, RydbergLindbladian]:
        ham = RydbergLindbladian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            pulser_linblads=pulser_lindblads,
            interaction_matrix=full_interaction_matrix,
            device=density_matrix.device,
        )

        def op(x: torch.Tensor) -> torch.Tensor:
            return -1j * dt * (ham @ x)

        return (
            krylov_exp(
                op,
                density_matrix,
                norm_tolerance=krylov_tolerance,
                exp_tolerance=krylov_tolerance,
                is_hermitian=False,
            ),
            ham,
        )
