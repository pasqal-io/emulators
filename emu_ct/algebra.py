from __future__ import annotations

import torch
from numbers import Number


def _add_factors(
    left: list[torch.tensor], right: list[torch.tensor]
) -> list[torch.tensor]:
    """
    Direct sum algorithm implementation to sum two tensor trains (MPS/MPO).
    It assumes the left and right bond are along the dimension 0 and 2 of each tensor.
    """
    num_sites = len(left)
    if num_sites != len(right):
        raise ValueError("Cannot sum two matrix products of different number of sites")

    new_tt = []
    for i, (core1, core2) in enumerate(zip(left, right)):
        core2 = core2.to(core1.device)
        if i == 0:
            core = torch.cat((core1, core2), dim=2)  # concatenate along the right bond
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
                (padded_c1, padded_c2), dim=2
            )  # concatenate along the right bond
        new_tt.append(core)
    return new_tt


def _mul_factors(factors: list[torch.tensor], scalar: Number) -> list[torch.tensor]:
    """
    Returns the tensor train (MPS/MPO) multiplied by a scalar.
    Assumes the orthogonal centre of the train is on the factor 0.
    """
    new_factors = [fac.clone() for fac in factors]
    new_factors[0] *= scalar
    return new_factors
