# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from emu_ct import MPS, MPO

import cupy as cp
import numpy as np

from cuquantum import cutensornet as cutn

def main():    
    print("cuTensorNet-vers:", cutn.get_version())
    dev = cp.cuda.Device()  # get current device
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print("===== device info ======")
    print("GPU-name:", props["name"].decode())
    print("GPU-clock:", props["clockRate"])
    print("GPU-memoryClock:", props["memoryClockRate"])
    print("GPU-nSM:", props["multiProcessorCount"])
    print("GPU-major:", props["major"])
    print("GPU-minor:", props["minor"])
    print("========================")

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
    print(mpo)
    out = mpo * mps
    print(out)
    

if __name__ == '__main__':
    main()