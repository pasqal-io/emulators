"""
Optional fused GPU kernels for `RydbergHamiltonian` (needs Triton).

The torch implementation applies H in N + 1 passes over the state: one for the
diagonal, then one `index_add_` per qubit. `apply` computes each output amplitude in
a single pass: it reads ψ at the amplitude and at its N single-bit-flip neighbours,
and writes the result once. The full diagonal (interaction, detuning and noise) is
built once per Hamiltonian by `make_diagonal`, so the loop over the neighbours only
does the off-diagonal terms.

The kernels are only used when Triton can be imported (it ships with the CUDA builds
of torch on Linux) and the state is a contiguous complex128 CUDA vector; otherwise
`RydbergHamiltonian` uses its torch implementation (CPU, macOS, Windows, no Triton).
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


def is_available(vec: torch.Tensor) -> bool:
    """Whether the fused kernel can be applied to `vec`."""
    return (
        triton is not None
        and vec.is_cuda
        and vec.dtype == torch.complex128
        and vec.dim() == 1
        and vec.is_contiguous()
        and vec.numel() >= 2**MIN_QUBITS
    )


if triton is not None:

    @triton.jit
    def _offsets(BLOCK: tl.constexpr, WIDE: tl.constexpr):
        pid = tl.program_id(0)
        if WIDE:
            pid = pid.to(tl.int64)
        return pid * BLOCK + tl.arange(0, BLOCK)

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
        offs = _offsets(BLOCK, WIDE)
        dr = tl.load(interaction_ptr + offs)
        di = tl.zeros_like(dr)
        for n in tl.static_range(NQUBITS):
            bit = (offs & (1 << (NQUBITS - 1 - n))) != 0
            dr += tl.where(bit, tl.load(tab_ptr + 8 * n + 2), tl.load(tab_ptr + 8 * n))
            if NOISE:
                di += tl.where(
                    bit, tl.load(tab_ptr + 8 * n + 3), tl.load(tab_ptr + 8 * n + 1)
                )
        if NOISE:
            pair = tl.arange(0, 2)
            tl.store(out_ptr + 2 * offs[:, None] + pair[None, :], tl.join(dr, di))
        else:
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

        Complex tensors are passed as interleaved float64 (torch.view_as_real) and
        read as [BLOCK, 2] blocks, so each amplitude is one 128-bit load.
        diag: full diagonal of H from `_diagonal_kernel`, complex only when NOISE.
        tab: 8 float64 per qubit n, (d0, d1, f0, f1) as (re, im) pairs: f_b multiplies
            x[k ^ bit_n] when bit_n(k) = b. Without COMPLEX, f0 = f1 is real.
        """
        offs = _offsets(BLOCK, WIDE)
        pair = tl.arange(0, 2)
        xr, xi = tl.split(tl.load(x_ptr + 2 * offs[:, None] + pair[None, :]))
        if NOISE:
            dr, di = tl.split(tl.load(diag_ptr + 2 * offs[:, None] + pair[None, :]))
            hr = dr * xr - di * xi
            hi = dr * xi + di * xr
        else:
            d = tl.load(diag_ptr + offs)
            hr = d * xr
            hi = d * xi
        for n in tl.static_range(NQUBITS):
            mask = 1 << (NQUBITS - 1 - n)
            j = offs ^ mask
            yr, yi = tl.split(tl.load(x_ptr + 2 * j[:, None] + pair[None, :]))
            if COMPLEX:
                bit = (offs & mask) != 0
                fr = tl.where(
                    bit, tl.load(tab_ptr + 8 * n + 6), tl.load(tab_ptr + 8 * n + 4)
                )
                fi = tl.where(
                    bit, tl.load(tab_ptr + 8 * n + 7), tl.load(tab_ptr + 8 * n + 5)
                )
                hr += fr * yr - fi * yi
                hi += fr * yi + fi * yr
            else:
                w = tl.load(tab_ptr + 8 * n + 4)
                hr += w * yr
                hi += w * yi
        s_r = tl.load(s_ptr)
        s_i = tl.load(s_ptr + 1)
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
    """
    nqubits = x.numel().bit_length() - 1
    noise = diag.is_complex()
    out = torch.empty_like(x)
    _apply_kernel[(x.numel() // BLOCK,)](
        torch.view_as_real(x),
        torch.view_as_real(out),
        torch.view_as_real(diag) if noise else diag,
        table,
        torch.view_as_real(factor),
        NQUBITS=nqubits,
        COMPLEX=is_complex,
        NOISE=noise,
        WIDE=nqubits >= WIDE_QUBITS,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return out
