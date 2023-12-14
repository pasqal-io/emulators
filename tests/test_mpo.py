# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from emu_ct import MPS, MPO

import cupy as cp
import numpy as np

def test_mul():    

    num_sites = 3
    max_virtual_extent = 12
    
    ##################################
    # Initialize an MPSHelper object
    ##################################

    mps = MPS(num_sites, max_virtual_extent)
    print(mps)
    factors = []
    for _ in range(num_sites):
        tensor = cp.zeros((1,2,2,1), dtype=np.complex128, order="F")
        tensor[0,0,1,0] = 1
        tensor[0,1,0,0] = 1
        factors.append(tensor)
    mpo = MPO(factors, max_virtual_extent)
    out = mpo * mps
    assert False