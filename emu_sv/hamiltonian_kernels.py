"""
Optional fused GPU kernel for `RydbergHamiltonian` (needs Triton).

The torch implementation applies H in N + 1 passes over the state: one for the
diagonal, then one `index_add_` per qubit. The kernel below computes each output
amplitude in a single pass: it reads ψ at the amplitude and at its N single-bit-flip
neighbours, and writes the result once. Applying H is memory-bound, so this is
roughly 10 times faster for large states.

The kernel is only used when Triton can be imported (it ships with the CUDA builds
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
    def _apply_kernel(
        x_ptr,
        out_ptr,
        diag_ptr,
        tab_ptr,
        s_ptr,
        NQUBITS: tl.constexpr,
        COMPLEX: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """
        out = s·(H x), with the complex scalar s read from s_ptr (float64 pair).

        Complex tensors are passed as interleaved float64 (torch.view_as_real).
        diag: real interaction diagonal ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ, length 2**NQUBITS.
        tab: 8 float64 per qubit n, (d0, d1, f0, f1) as (re, im) pairs: d_b is added
            to the diagonal and f_b multiplies x[k ^ bit_n] when bit_n(k) = b.
        """
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        xr = tl.load(x_ptr + 2 * offs)
        xi = tl.load(x_ptr + 2 * offs + 1)
        d = tl.load(diag_ptr + offs)
        hr = d * xr
        hi = d * xi
        for n in tl.static_range(NQUBITS):
            mask = 1 << (NQUBITS - 1 - n)
            bit = (offs & mask) != 0
            j = offs ^ mask
            yr = tl.load(x_ptr + 2 * j)
            yi = tl.load(x_ptr + 2 * j + 1)
            ddr = tl.where(bit, tl.load(tab_ptr + 8 * n + 2), tl.load(tab_ptr + 8 * n))
            if COMPLEX:
                ddi = tl.where(
                    bit, tl.load(tab_ptr + 8 * n + 3), tl.load(tab_ptr + 8 * n + 1)
                )
                fr = tl.where(
                    bit, tl.load(tab_ptr + 8 * n + 6), tl.load(tab_ptr + 8 * n + 4)
                )
                fi = tl.where(
                    bit, tl.load(tab_ptr + 8 * n + 7), tl.load(tab_ptr + 8 * n + 5)
                )
                hr += ddr * xr - ddi * xi + fr * yr - fi * yi
                hi += ddr * xi + ddi * xr + fr * yi + fi * yr
            else:
                w = tl.load(tab_ptr + 8 * n + 4)
                hr += ddr * xr + w * yr
                hi += ddr * xi + w * yi
        s_r = tl.load(s_ptr)
        s_i = tl.load(s_ptr + 1)
        tl.store(out_ptr + 2 * offs, s_r * hr - s_i * hi)
        tl.store(out_ptr + 2 * offs + 1, s_r * hi + s_i * hr)


def apply(
    x: torch.Tensor,
    diag: torch.Tensor,
    table: torch.Tensor,
    factor: torch.Tensor,
    is_complex: bool,
) -> torch.Tensor:
    """
    Return factor·(H x) for the Hamiltonian described by `diag` and `table`.
    `factor` is a one-element complex128 tensor on the device of `x` (a tensor rather
    than a Python number, which Triton would pass as float32).
    """
    out = torch.empty_like(x)
    _apply_kernel[(x.numel() // BLOCK,)](
        torch.view_as_real(x),
        torch.view_as_real(out),
        diag,
        table,
        torch.view_as_real(factor),
        NQUBITS=x.numel().bit_length() - 1,
        COMPLEX=is_complex,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return out
