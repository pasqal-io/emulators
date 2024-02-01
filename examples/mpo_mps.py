from emu_ct import MPS, MPO, inner
import torch

num_sites = 3

mps = MPS(num_sites)
print("MPS:", mps)
factors = []
for _ in range(num_sites):
    tensor = torch.zeros(1, 2, 2, 1, dtype=torch.complex128)
    tensor[0, 0, 1, 0] = 1
    tensor[0, 1, 0, 0] = 1
    factors.append(tensor)
mpo = MPO(factors)
print("MPO;", mpo)
out = mpo * mps
print("MPS*MPO:", out)
assert inner(out, out) == 1.0 + 0.0j, "<111|111> = 1"
assert inner(mps, out) == 0.0 + 0.0j, "<000|111> = 0"
