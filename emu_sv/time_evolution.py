import torch
from typing import Any, no_type_check
from emu_base.math.krylov_exp import krylov_exp
from emu_base.math.double_krylov import double_krylov
from emu_sv.hamiltonian import RydbergHamiltonian


class DHDOmegaSparse:
    """Implementation of derivative of the Rydberg Hamiltonian respect to Omega
    following the RydbergHamiltonian sparse format. At the end we are doing,
    - i dt dH/d𝛺ₖ =− i dt 1/2 𝜎ₓᵏ"""

    def __init__(self, dt: int, index: int, device: str, nqubits: int):
        self.index = index
        self.nqubits = nqubits
        self.dt = dt
        self.inds = torch.tensor([1, 0], device=device)  # flips the state, for 𝜎ₓ

    def __matmul__(self, vec: torch.Tensor) -> torch.Tensor:
        vec = vec.reshape((vec.shape[0],) + (2,) * self.nqubits)  # add batch dimension
        result = torch.zeros(vec.shape, device=vec.device, dtype=vec.dtype)
        result.index_add_(self.index + 1, self.inds, vec, alpha=-1j * self.dt / 2)
        return result.reshape(vec.shape[0], -1)


class DHDDeltaSparse:
    """Implementation of derivative of the Rydberg Hamiltonian respect to Omega
    following the RydbergHamiltonian sparse format. At the end we are doing,
    - i dt dH/d𝛺ₖ =− i dt 1/2 𝜎ₓᵏ"""

    def __init__(self, dt: int, index: int, device: str, nqubits: int):
        self.index = index
        self.nqubits = nqubits
        diag = torch.zeros((2,) * nqubits, dtype=torch.complex128, device=device)
        i_fixed = diag.select(index, 1)
        i_fixed += 1j * dt  # add the delta term for this qubit
        self.diag = diag.reshape(-1)

    def __matmul__(self, vec: torch.Tensor) -> torch.Tensor:
        return vec * self.diag


class EvolveStateVector(torch.autograd.Function):
    """Custom autograd implementation of a step in the time evolution."""

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
        under the Hamiltonian H built from the input Tensor parameters, omega, delta and
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
        ham = RydbergHamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            interaction_matrix=interaction_matrix,
            device=state.device,
        )
        op = lambda x: -1j * dt * (ham * x)
        res = krylov_exp(
            op,
            state,
            norm_tolerance=krylov_tolerance,
            exp_tolerance=krylov_tolerance,
            is_hermitian=True,
        )
        ctx.save_for_backward(omegas, deltas, phis, interaction_matrix, state)
        ctx.dt = dt
        ctx.tolerance = krylov_tolerance
        return res, ham

    @no_type_check  # mypy complains and I don't know why
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
        and return the gradients of the loss with respect to the input tensors
            - gΩⱼ = ∂L/∂Ωⱼ =〈gψ(t+dt)|DU(H,∂H/∂Ωⱼ)|ψ(t)〉
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
            〈gψ(t+dt)|d|ψ(t+dt)〉= ∑ⱼgΔⱼ*dΔⱼ + ∑ⱼgΩⱼ*dΩⱼ +〈gψ(t)|d|ψ(t)〉  (1)

        For the exponential map U = exp(-i dt H), differentiating reads:
            d|ψ(t+dt)〉= dU|ψ(t)〉+ Ud|ψ(t)〉
            dU = ∑ⱼdU(H,∂H/∂Δⱼ) + ∑ⱼdU(H,∂H/∂Ωⱼ)                        (2)

        where dU(H,dH) is the Fréchet derivative of the exponential map
        along the direction dH
        - https://eprints.maths.manchester.ac.uk/1218/1/covered/MIMS_ep2008_26.pdf
        - https://en.wikipedia.org/wiki/Derivative_of_the_exponential_map

        Substituting (2) into (1) leads to the expressions of the gradients
        with respect to the input tensors above.

        Variations with respect to the Hamiltonian parameters are computed as
            gΩ = 〈gψ(t+dt)|dU(H,∂H/∂Ω)|ψ(t)〉
               = Tr( ∂H/∂Ω @ dU(H,|ψ(t)〉〈gψ(t+dt)|) ),
        where under the trace sign, ∂H/∂Ω and |ψ(t)〉〈gψ(t+dt)| can be switched.

        - The Fréchet derivative is computed in a Arnoldi-Gram-Schmidt
        decomposition in the `double_krylov` method:
            dU(H,|a〉〈b|) = Va @ dS @ Vb*
        where V_odd and V_even are orthogonal Krylov basis associated
        with |a〉and |b〉respectively.

        - The action of the derivatives of the Hamiltonian with
        respect to the input parameters are implemented separately in
            - ∂H/∂Ω: `DHDOmegaSparse`
            - ∂H/∂Δ: `DHDDeltaSparse`


        - dU is the Fréchet derivative
        F_exp(-i dt H, dH)
        exp([H dH])  = (exp(H) L(H,dH))
           ([0  H])    (  0     exp(H))

        For the Hamiltonian parameters
            gΩ = 〈gψ(t+dt)|L(H,∂H/∂Ω)|ψ(t)〉
               = Tr( ∂H/∂Ω @ L(H,|ψ(t)〉〈gψ(t+dt)|) )

        L(H,|ψ(t)〉〈gψ(t+dt)|) = VTV^{-1}

        Tr( |0     0| @ exp|dt*H  |ψ(t)〉〈gψ(t+dt)| | )
          ( |∂H/∂Ω 0|      |  0          dt*H       | )


        """
        omegas, deltas, phis, interaction_matrix, state = ctx.saved_tensors
        dt = ctx.dt
        tolerance = ctx.tolerance
        ham = RydbergHamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            interaction_matrix=interaction_matrix,
            device=state.device,
        )

        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            op = lambda x: -1j * dt * (ham * x)
            lanczos_vectors_even, dS, lanczos_vectors_odd = double_krylov(
                op, state, grad_state_out, tolerance
            )
            even_block = torch.stack(lanczos_vectors_even)
            odd_block = torch.stack(lanczos_vectors_odd)
            e_l = torch.tensordot(dS, even_block, dims=([0], [0]))

        grad_omegas, grad_deltas, grad_phis, grad_state_in = None, None, None, None

        if ctx.needs_input_grad[1]:
            grad_omegas = torch.zeros_like(omegas)
            for i in range(omegas.shape[-1]):
                # dh as per the docstring
                dho = DHDOmegaSparse(dt, i, e_l.device, omegas.shape[-1])
                # compute the trace
                v = dho @ e_l
                grad_omegas[i] = torch.tensordot(
                    odd_block.conj(), v, dims=([0, 1], [0, 1])
                ).real

        if ctx.needs_input_grad[2]:
            grad_deltas = torch.zeros(deltas.shape, dtype=deltas.dtype)
            for i in range(deltas.shape[-1]):
                # dh as per the docstring
                dhd = DHDDeltaSparse(dt, i, e_l.device, deltas.shape[-1])
                # compute the trace
                v = dhd @ e_l
                grad_deltas[i] = torch.tensordot(
                    odd_block.conj(), v, dims=([0, 1], [0, 1])
                ).real

        if ctx.needs_input_grad[3]:
            grad_phis = torch.zeros(phis.shape, dtype=phis.dtype)

        if ctx.needs_input_grad[5]:
            op = lambda x: (1j * dt) * (ham * x)
            grad_state_in = krylov_exp(op, grad_state_out, tolerance, tolerance)

        return None, grad_omegas, grad_deltas, grad_phis, None, grad_state_in, None
