"""
This module contains the different permute functions that are used in the context
of the MPS backend.
"""

from collections import Counter

import torch

from pulser.backend import Results
import emu_mps.optimatrix as optimat


# pylint: disable=protected-access

#TODO check if it is possible to avoid protected accesses from objects (e.g. results._results)

#TODO correct docstrings


def permute_bitstrings(results: Results, perm: torch.Tensor) -> None:
    """
    Bitstring permutator functions

    Args:

        results (Results): Result data
        perm (torch.Tensor): Pytorch tensor that
    """
    if "bitstrings" not in results.get_result_tags():
        return
    uuid_bs = results._find_uuid("bitstrings")

    results._results[uuid_bs] = [
        Counter({optimat.permute_string(bstr, perm): c for bstr, c in bs_counter.items()})
        for bs_counter in results._results[uuid_bs]
    ]


def permute_occupations_and_correlations(results: Results, perm: torch.Tensor) -> None:
    """
    Function permuting the occupations and correlations

    Args:
        results (Results): Results data structure
        perm (torch.Tensor): Pytorch permutation matrix
    """
    for corr in ["occupation", "correlation_matrix"]:
        if corr not in results.get_result_tags():
            continue

        uuid_corr = results._find_uuid(corr)
        corrs = results._results[uuid_corr]
        results._results[uuid_corr] = (
            [  # vector quantities become lists after results are serialized (e.g. for checkpoints)
                optimat.permute_tensor(
                    corr if isinstance(corr, torch.Tensor) else torch.tensor(corr), perm
                )
                for corr in corrs
            ]
        )


def permute_atom_order(results: Results, perm: torch.Tensor) -> None:
    """
    Function permuting the atom order

    Args:

        results (Results): Results data structure
        perm (torch.Tensor): Pytorch permutation matrix
    """
    at_ord = list(results.atom_order)
    at_ord = optimat.permute_list(at_ord, perm)
    results.atom_order = tuple(at_ord)
