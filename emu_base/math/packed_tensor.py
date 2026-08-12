import torch


class PackedHermitianTensor:
    """
    Pack a tensor of shape (χ, m, χ), Hermitian in axes 0 and 2,
    into shape (χ(χ+1)/2, m) by storing the lower triangle
    of each (χ, χ) slice at fixed middle index m.
    The `PackedHermitianTensor` is used to represent left and right
    baths nodes in TDVP/DMRG algorithms.
    """

    def __init__(
        self,
        chi: int,
        m: int,
        *,
        check_hermitian: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> None:
        self._packed_data = torch.zeros(
            int(chi * (chi + 1) / 2), m, dtype=torch.complex128
        )
        self.check_hermitian = check_hermitian
        self.rtol = rtol
        self.atol = atol
        self.chi = chi
        self.m = m

    def pack(self, h: torch.Tensor) -> None:
        if h.ndim != 3 or h.shape[0] != h.shape[2]:
            raise ValueError(f"Expected shape (χ, m, χ), got {tuple(h.shape)}")
        if self.chi != h.shape[0] or self.m != h.shape[1]:
            raise ValueError(
                f"Initialized for ({self.chi},{self.m},{self.chi}), got {h.shape}"
            )

        if self.check_hermitian and not torch.allclose(
            h, h.transpose(0, 2).conj(), rtol=self.rtol, atol=self.atol
        ):
            raise ValueError("Tensor is not Hermitian in axes 0 and 2")

        self._ii, self._kk = torch.tril_indices(self.chi, self.chi, device=h.device)
        self._packed_data[:, :] = h[self._ii, :, self._kk]
        self._packed_data = self._packed_data.to(h.device)

    def unpack(self) -> torch.Tensor:
        vals = self._packed_data
        m = vals.shape[1]

        h = torch.zeros(
            (self.chi, m, self.chi),
            dtype=self._packed_data.dtype,
            device=self._packed_data.device,
        )
        h[self._ii, :, self._kk] = vals

        offdiag = self._ii != self._kk
        h[self._kk[offdiag], :, self._ii[offdiag]] = vals[offdiag].conj()
        return h
