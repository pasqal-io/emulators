import torch


class PackedHermitianTensor:
    """
    Pack a tensor of shape (χ, m, χ), Hermitian in axes 0 and 2,
    into shape (m, χ(χ+1)//2) by storing the lower triangle
    of each (χ, χ) slice at fixed middle index.
    The `PackedHermitianTensor` is used to represent left and right
    baths in TDVP/DMRG algorithms.

    Lower-triangular packed order:
        (0,0),
        (1,0), (1,1),
        (2,0), (2,1), (2,2),
        ...
    """

    def __init__(
        self,
        h: torch.Tensor,
        *,
        check_hermitian: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> None:
        if h.ndim != 3 or h.shape[0] != h.shape[2]:
            raise ValueError(f"Expected shape (χ, m, χ), got {tuple(h.shape)}")

        if check_hermitian and not torch.allclose(
            h, h.transpose(0, 2).conj(), rtol=rtol, atol=atol
        ):
            raise ValueError("Tensor is not Hermitian in axes 0 and 2")

        self.chi = h.shape[0]
        self._ii, self._kk = torch.tril_indices(self.chi, self.chi, device=h.device)
        self._packed_data = h[self._ii, :, self._kk].transpose(0, 1).contiguous()

    def unpack(self) -> torch.Tensor:
        m = self._packed_data.shape[0]
        vals = self._packed_data.transpose(0, 1)

        h = torch.zeros(
            (self.chi, m, self.chi),
            dtype=self._packed_data.dtype,
            device=self._packed_data.device,
        )
        h[self._ii, :, self._kk] = vals

        offdiag = self._ii != self._kk
        h[self._kk[offdiag], :, self._ii[offdiag]] = vals[offdiag].conj()
        return h
