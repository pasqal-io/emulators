# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from emu_ct import MPS, MPO, OperatorString
import torch


def test_mul():

    num_sites = 3

    mps = MPS(num_sites)
    factors = []
    for _ in range(num_sites):
        tensor = torch.zeros(1, 2, 2, 1, dtype=torch.complex128)
        tensor[0, 0, 1, 0] = 1
        tensor[0, 1, 0, 0] = 1
        factors.append(tensor)
    mpo = MPO(factors)
    out = mpo * mps
    for i in out.factors:
        assert torch.allclose(
            i, torch.tensor([[[0], [1]]], dtype=torch.complex128, device=i.device)
        )
    out = mps * mpo
    for i in out.factors:
        assert torch.allclose(
            i, torch.tensor([[[0], [1]]], dtype=torch.complex128, device=i.device)
        )


def test_from_operator_string():
    x = OperatorString([1.0, 1.0], ["sigma_gr", "sigma_rg"])
    z = OperatorString([1.0, -1.0], ["sigma_gg", "sigma_rr"])
    operators = {"X": x, "Z": z}
    operations = [
        (OperatorString([2.0], ["X"]), ["q0", "q2"]),
        (OperatorString([3], ["Z"]), ["q1"]),
    ]
    mpo = MPO.from_operator_string(("r", "g"), ["q0", "q1", "q2"], operations, operators)
    assert torch.allclose(
        mpo.factors[0],
        torch.tensor(
            [[0.0, 2.0], [2.0, 0.0]], dtype=torch.complex128, device=mpo.factors[0].device
        ).reshape(1, 2, 2, 1),
    )
    assert torch.allclose(
        mpo.factors[1],
        torch.tensor(
            [[3.0, 0.0], [0.0, -3.0]],
            dtype=torch.complex128,
            device=mpo.factors[1].device,
        ).reshape(1, 2, 2, 1),
    )
    assert torch.allclose(
        mpo.factors[2],
        torch.tensor(
            [[0.0, 2.0], [2.0, 0.0]], dtype=torch.complex128, device=mpo.factors[2].device
        ).reshape(1, 2, 2, 1),
    )
