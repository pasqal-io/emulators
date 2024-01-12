# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from emu_ct import MPS, MPO, Config
import torch


def test_mul():

    num_sites = 3
    Config().set_num_devices_to_use(0)

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
        assert torch.allclose(i, torch.tensor([[[0], [1]]], dtype=torch.complex128))
    out = mps * mpo
    for i in out.factors:
        assert torch.allclose(i, torch.tensor([[[0], [1]]], dtype=torch.complex128))
