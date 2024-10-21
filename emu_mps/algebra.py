from __future__ import annotations

import torch
import math
from emu_mps.utils import truncate_impl


def add_factors(
    left: list[torch.tensor], right: list[torch.tensor]
) -> list[torch.tensor]:
    """
    Direct sum algorithm implementation to sum two tensor trains (MPS/MPO).
    It assumes the left and right bond are along the dimension 0 and -1 of each tensor.
    """
    num_sites = len(left)
    if num_sites != len(right):
        raise ValueError("Cannot sum two matrix products of different number of sites")

    new_tt = []
    for i, (core1, core2) in enumerate(zip(left, right)):
        core2 = core2.to(core1.device)
        if i == 0:
            core = torch.cat((core1, core2), dim=-1)  # concatenate along the right bond
        elif i == (num_sites - 1):
            core = torch.cat((core1, core2), dim=0)  # concatenate along the left bond
        else:
            pad_shape_1 = (core2.shape[0], *core1.shape[1:])
            padded_c1 = torch.cat(
                (
                    core1,
                    torch.zeros(pad_shape_1, device=core1.device, dtype=core1.dtype),
                ),
                dim=0,  # concatenate along the left bond
            )
            pad_shape_2 = (core1.shape[0], *core2.shape[1:])
            padded_c2 = torch.cat(
                (
                    torch.zeros(pad_shape_2, device=core1.device, dtype=core1.dtype),
                    core2,
                ),
                dim=0,  # concatenate along the left bond
            )
            core = torch.cat(
                (padded_c1, padded_c2), dim=-1
            )  # concatenate along the right bond
        new_tt.append(core)
    return new_tt


def scale_factors(
    factors: list[torch.tensor], scalar: complex, *, which: int
) -> list[torch.tensor]:
    """
    Returns a new list of factors where the tensor at the given index is scaled by `scalar`.
    """
    return [scalar * f if i == which else f for i, f in enumerate(factors)]


def zip_right_step(
    slider: torch.tensor,
    top: torch.tensor,
    bottom: torch.tensor,
) -> torch.tensor:
    """
    Returns a new `MPS/O` factor of the result of the multiplication MPO @ MPS/O,
    and the updated slider, performing a single step of the
    [zip-up algorithm](https://tensornetwork.org/mps/algorithms/zip_up_mpo/).

    Args:
    - `slider`: utility tensor for the zip-up algorithm.
    - `top`: factor of the applied MPO.
    - `bottom`: factor of the MPS/O to which the MPO is being applied.

    First, moves all tensors to `bottom.device`.
    Second, it contracts `top` and then `bottom` to `slider`.
    The resulting tensor is then QR factorized into a
    new factor and the updated slider for the next zip step.

    Note:
    The method assumes that:
        - `top` is a valid MPO factor of shape
    (left_link_dim, out_site_dim, in_site_dim, right_link_dim).
        - `bottom` is a valid MPO/S factor
    """
    if slider.shape[1:] != (top.shape[0], bottom.shape[0]):
        msg = (
            f"Contracted dimensions between the slider, {slider.shape[1:]} on dims 1 and 2, "
            f"and the two factors, {(top.shape[0], bottom.shape[0])} on dim 0, need to match."
        )
        raise ValueError(msg)

    slider = slider.to(bottom.device)
    top = top.to(bottom.device)

    # merge top and bottom into slider
    slider = torch.tensordot(slider, top, dims=([1], [0]))
    slider = torch.tensordot(slider, bottom, dims=([3, 1], [1, 0]))

    if len(bottom.shape) == 4:  # MPO factor
        slider = slider.transpose(2, 3)

    # reshape slider as matrix
    left_inds = (slider.shape[0], *bottom.shape[1:-1])
    right_inds = (top.shape[-1], bottom.shape[-1])
    slider = slider.reshape(math.prod(left_inds), math.prod(right_inds))

    L, slider = torch.linalg.qr(slider)

    # reshape slider to its original shape
    slider = slider.reshape((-1, *right_inds))
    # reshape left as MPS/O factor and
    return L.reshape(*left_inds, -1), slider


def zip_right(
    top_factors: list[torch.tensor],
    bottom_factors: list[torch.tensor],
    max_error: float = 1e-5,
    max_rank: int = 1024,
) -> list[torch.tensor]:
    """
    Returns a new matrix product, resulting from applying `top` to `bottom`.
    The resulting factors are:
     - of the same order as `bottom` factors
     - on the same device of `bottom` factors
     - orthogonalized on the first element
     - truncated to `max_error`/`max_rank`

    Args:
    - `top`: MPO factors to be applied.
    - `bottom`: MPS/O factors to which the MPO factors are being applied.

    Note:
        Implements a general [zip-up](https://tensornetwork.org/mps/algorithms/zip_up_mpo/)
        algorithm for applying MPO factors to both MPO and MPS factors.
        A final truncation sweep, from right to left,
        moves back the orthogonal center to the first element.
    """
    if len(top_factors) != len(bottom_factors):
        raise ValueError("Cannot multiply two matrix products of different lengths.")

    slider = torch.ones(1, 1, 1, dtype=torch.complex128)
    new_factors = []
    for top, bottom in zip(top_factors, bottom_factors):
        res, slider = zip_right_step(slider, top, bottom)
        new_factors.append(res)
    new_factors[-1] @= slider[:, :, 0]

    truncate_impl(new_factors, max_error=max_error, max_rank=max_rank)

    return new_factors
