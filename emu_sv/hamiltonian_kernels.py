"""
Optional fused GPU kernels for `RydbergHamiltonian` (needs Triton).

The torch implementation applies H in N + 1 passes over the state: one for the
diagonal, then one `index_add_` per qubit. `apply` computes each output amplitude in
a single pass: it reads ψ at the amplitude and at its N single-bit-flip neighbours,
and writes the result once. The full diagonal (interaction, detuning and noise) is
built once per Hamiltonian by `make_diagonal`, so the loop over the neighbours only
does the off-diagonal terms.

The kernels are only used when Triton can be imported (it ships with the CUDA builds
of torch on Linux) and the state is a complex128 CUDA vector; otherwise
`RydbergHamiltonian` uses its torch implementation (CPU, macOS, Windows, no Triton).

Notes on the triton programming model:

On a CUDA gpu, each parallel execution of the code is called a thread, and threads are
grouped in blocks. Each thread must execute exactly the same instructions but
potentially on different data. Similar to a broadcasting statement in numpy, writing
c=a+b for two arrays does many single adds, but the two input elements are at a different
location and so is the output. Consequently, in triton each CUDA block is called a
program and you write each program using broadcast semantics on a vector.
You launch a kernel as `kernel[(n_blocks_to_use,)]` and then get the program id at
runtime using `tl.program_id`. From this you can compute what subset of the data
that program needs to operate on.
"""

# Triton kernels cannot be type-annotated in a way mypy understands.
# mypy: disable-error-code="no-untyped-def"

import torch

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:
    triton = None

BLOCK = 256
# Below this many qubits the torch path is launch-bound and already cheap.
MIN_QUBITS = 8
# From this many qubits, offsets into the interleaved (re, im) float64 view
# no longer fit in 32 bits.
WIDE_QUBITS = 30


def is_available(device: torch.device, nqubits: int) -> bool:
    """Whether the fused kernel can apply H to complex128 states of `nqubits` qubits
    on `device`."""
    return triton is not None and device.type == "cuda" and nqubits >= MIN_QUBITS


if triton is not None:

    @triton.jit
    def _offsets(BLOCK: tl.constexpr, WIDE: tl.constexpr):
        """
        Triton sees an entire block of cuda threads as a single vectorized program.
        This returns a vector of all the indices in out handled by each thread
        in the block.
        """
        pid = tl.program_id(0)
        if WIDE:
            pid = pid.to(tl.int64)  # this triggers if n_qubits >= WIDE_QUBITS
        return pid * BLOCK + tl.arange(0, BLOCK)  # blockIdx.x * blockDim.x + threadIdx.x

    @triton.jit
    def _diagonal_kernel(
        interaction_ptr,
        tab_ptr,
        out_ptr,
        NQUBITS: tl.constexpr,
        NOISE: tl.constexpr,
        WIDE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """
        out = interaction + ∑ₙ d_b(n), with d_b(n) the (re, im) pair at tab[8n + 2b]
        for b = bit_n(k). out is real, or interleaved complex when NOISE.
        """
        offs = _offsets(BLOCK, WIDE)  # indices of out this program works on

        # load the corresponding interaction terms in registers
        dr = tl.load(interaction_ptr + offs)  # load the corresponding interaction terms
        di = tl.zeros_like(dr)  # removed by jit compiler if NOISE=False

        # for qubit n, add the diagonal part of H corresponding to that qubit
        for n in tl.static_range(NQUBITS):  # static range loops are unrolled
            # which element of the 2x2 matrix to load depends on the sign of bit n
            bit = (offs & (1 << (NQUBITS - 1 - n))) != 0

            # add the delta term to the interaction
            dr += tl.where(bit, tl.load(tab_ptr + 8 * n + 2), tl.load(tab_ptr + 8 * n))
            if NOISE:  # only add the imaginary part if there is noise
                di += tl.where(
                    bit, tl.load(tab_ptr + 8 * n + 3), tl.load(tab_ptr + 8 * n + 1)
                )
        if NOISE:  # output is imaginary, pack the results and store
            pair = tl.arange(0, 2)
            tl.store(out_ptr + 2 * offs[:, None] + pair[None, :], tl.join(dr, di))
        else:  # output is real, simply store the result
            tl.store(out_ptr + offs, dr)

    @triton.jit
    def _apply_kernel(
        x_ptr,
        out_ptr,
        diag_ptr,
        tab_ptr,
        s_ptr,
        NQUBITS: tl.constexpr,
        COMPLEX: tl.constexpr,
        NOISE: tl.constexpr,
        WIDE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """
        out = s·(H x), with the complex scalar s read from s_ptr (float64 pair).

        Complex tensors are passed as interleaved float64
        (i.e. of shape (...,2) where the last index represents real and imag part)
        and read as [BLOCK, 2] blocks, so each amplitude is one 128-bit load.
        diag: full diagonal of H from `_diagonal_kernel`, complex only when NOISE.
        tab: 8 float64 per qubit n, (d0, d1, f0, f1) as (re, im) pairs: f_b multiplies
            x[k ^ bit_n] when bit_n(k) = b. Without COMPLEX, f0 = f1 is real.
        """
        offs = _offsets(BLOCK, WIDE)  # the indices of out this program will operate on
        pair = tl.arange(0, 2)  # for indexing real and imaginary parts

        # load the part of x needed by this program for applying the diagonal of H
        # real and imaginary part are treated separately.
        xr, xi = tl.split(tl.load(x_ptr + 2 * offs[:, None] + pair[None, :]))

        # apply the diagonal to x, store in a register h
        if NOISE:  # diagonal is imaginary, so handle both real and imag parts
            dr, di = tl.split(tl.load(diag_ptr + 2 * offs[:, None] + pair[None, :]))
            hr = dr * xr - di * xi
            hi = dr * xi + di * xr
        else:  # diag is real. The branch is chosen at compile time by the jit compiler
            d = tl.load(diag_ptr + offs)
            hr = d * xr
            hi = d * xi

        # this does all the _index_add operations from RydbergHamiltonian._apply_torch
        # the loop is executed on the device, leading to speedup
        # for qubit n, it applies the corresponding single-qubit term in H
        for n in tl.static_range(NQUBITS):
            # x[j] ends up in out[offs] under the action of the n-th term
            mask = 1 << (NQUBITS - 1 - n)
            j = offs ^ mask

            # load the real and imag part of x[j] separately
            yr, yi = tl.split(tl.load(x_ptr + 2 * j[:, None] + pair[None, :]))
            if COMPLEX:  # omega complex, handle real and imag parts separately
                # which off-diagonal element of the 2x2 matrix to take.
                # depends on the sign of bit n
                bit = (offs & mask) != 0

                # load the real and imaginary part, conditional on the sign
                fr = tl.where(
                    bit, tl.load(tab_ptr + 8 * n + 6), tl.load(tab_ptr + 8 * n + 4)
                )
                fi = tl.where(
                    bit, tl.load(tab_ptr + 8 * n + 7), tl.load(tab_ptr + 8 * n + 5)
                )

                hr += fr * yr - fi * yi  # apply the matrix element and flip the bit
                hi += fr * yi + fi * yr
            else:  # omega is real, only apply omega_n/2 X_n
                w = tl.load(tab_ptr + 8 * n + 4)  # f0_n = omega_n/2
                hr += w * yr  # scale and flip the bit, increment the register
                hi += w * yi

        # load the scalar prefactor (e.g. -1.j*dt)
        s_r = tl.load(s_ptr)
        s_i = tl.load(s_ptr + 1)

        # scale the register h with s, and store in out
        tl.store(
            out_ptr + 2 * offs[:, None] + pair[None, :],
            tl.join(s_r * hr - s_i * hi, s_r * hi + s_i * hr),
        )


def make_diagonal(
    interaction_diag: torch.Tensor, table: torch.Tensor, noise: bool
) -> torch.Tensor:
    """
    Return the full diagonal of H for `apply`: the real interaction part plus the
    per-qubit detuning (and noise) terms of `table`. Real, or complex128 with `noise`.
    """
    nqubits = interaction_diag.numel().bit_length() - 1
    out = torch.empty(
        interaction_diag.numel(),
        dtype=torch.complex128 if noise else torch.float64,
        device=interaction_diag.device,
    )
    _diagonal_kernel[(interaction_diag.numel() // BLOCK,)](
        interaction_diag,
        table,
        torch.view_as_real(out) if noise else out,
        NQUBITS=nqubits,
        NOISE=noise,
        WIDE=nqubits >= WIDE_QUBITS,
        BLOCK=BLOCK,
        num_warps=8,  # BLOCK/32 since 32 threads in a warp
    )
    return out


def apply(
    x: torch.Tensor,
    diag: torch.Tensor,
    table: torch.Tensor,
    factor: torch.Tensor,
    is_complex: bool,
) -> torch.Tensor:
    """
    Return factor·(H x) for the Hamiltonian described by `diag` (from
    `make_diagonal`) and `table`.
    `factor` is a one-element complex128 tensor on the device of `x` (a tensor rather
    than a Python number, which Triton would pass as float32).

    `table` is a 1D float64 tensor of length 8·N on the device of `x`, built by
    `RydbergHamiltonian._kernel_args`. It holds four complex coefficients per qubit n
    (n = 0 is the most significant bit of the basis index), each stored as
    (real, imaginary):

        index        coefficient                            used
        8n, 8n+1     d0 = noise[0, 0]  (0 without noise)    diagonal, bit n = 0
        8n+2, 8n+3   d1 = -Δₙ + noise[1, 1]                 diagonal, bit n = 1
        8n+4, 8n+5   f0 = Ωₙ/2·exp(-iφₙ) + noise[0, 1]      x[k ^ bit_n], bit n of k = 0
        8n+6, 8n+7   f1 = Ωₙ/2·exp(+iφₙ) + noise[1, 0]      x[k ^ bit_n], bit n of k = 1

    Ωₙ, Δₙ and φₙ are the Rabi frequency, detuning and phase of qubit n, and `noise`
    is the 2x2 single-qubit term -½i∑ⱼLⱼ†Lⱼ (see `compute_noise_from_lindbladians`).
    f0 and f1 are the matrix elements ⟨0|H|1⟩ and ⟨1|H|0⟩ of qubit n.
    d0 and d1 are only read by `make_diagonal`, which adds them to the interaction
    diagonal; `apply` reads f0 and f1. Without `is_complex` (no phase, no noise),
    f0 = f1 = Ωₙ/2 is real and only index 8n+4 is read.
    """
    assert len(x.shape) == 1
    assert x.shape == diag.shape
    x = x.contiguous()
    nqubits = x.numel().bit_length() - 1
    assert table.shape[0] // 8 == nqubits
    noise = diag.is_complex()
    out = torch.empty_like(x)
    _apply_kernel[(x.numel() // BLOCK,)](
        torch.view_as_real(x),  # split the real and imaginary parts with a separate index
        torch.view_as_real(out),
        torch.view_as_real(diag) if noise else diag,
        table,
        torch.view_as_real(factor),
        NQUBITS=nqubits,
        COMPLEX=is_complex,
        NOISE=noise,
        WIDE=nqubits >= WIDE_QUBITS,
        BLOCK=BLOCK,
        num_warps=8,  # BLOCK/32 since 32 threads in a warp
    )
    return out
