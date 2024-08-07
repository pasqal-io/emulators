import torch

from emu_mps import MPO, MPS, inner
from emu_mps.tdvp import evolve_tdvp


num_sites = 3

mps = MPS(num_sites)
print("MPS:", mps)

# X1+X2+X3
mpo_factor1 = torch.tensor([[[[0, 1], [1, 0]], [[1, 0], [0, 1]]]], dtype=torch.complex128)
mpo_factor2 = torch.tensor(
    [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]],
    dtype=torch.complex128,
)
mpo_factor3 = torch.tensor(
    [[[[1], [0]], [[0], [1]]], [[[0], [1]], [[1], [0]]]], dtype=torch.complex128
)
mpo = MPO([mpo_factor1, mpo_factor2, mpo_factor3])
print("MPO;", mpo)

out = mpo * mps
print("MPO*MPS:", out)
assert inner(out, out) == 3.0 + 0.0j, "<110+101+011=|110+101+011> = 3"
assert inner(mps, out) == 0.0 + 0.0j, "<000|110+101+011> = 0"

evolve_tdvp(-0.5j * torch.pi, mps, mpo, mps.precision)
print("exp(-i pi MPO / 2)*MPS:", mps)
assert abs(inner(mps, mps) - 1) < 1e-8, "<-i*111|-i*111> = 1"
assert abs(inner(mps, out)) < 1e-8, "<-i*111|110+101+011> = 0"
