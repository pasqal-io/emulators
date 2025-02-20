import torch

torch.manual_seed(1337)

dtype = torch.complex128
device = "cpu"

i = 1
j = 3

N = 5
A = torch.randn((2,) * N, dtype=dtype, device=device)
B = A.clone()

# for i in range(N):

i_fixed = A.select(dim=i, index=1)
sh = (2**i, 2, 2 ** (j - i - 1), 2, 2 ** (N - j - 1))
B = B.reshape(sh)
tmp_i_fixed = B.select(dim=1, index=1)
tmp_i_fixed = tmp_i_fixed.reshape((2**i, 2 ** (j - i - 1), 2, 2 ** (N - j - 1)))
# print(A.reshape((2,)*N))
assert torch.allclose(i_fixed, tmp_i_fixed.reshape((2,) * (N - 1)))
# for j in range(i+1, N):

i_j_fixed = i_fixed.select(j - 1, 1)


tmp_i_j_fixed = tmp_i_fixed.select(dim=2, index=1)
# print("expected\n", i_j_fixed)
# print("got\n", tmp_i_j_fixed)
assert torch.allclose(i_j_fixed, tmp_i_j_fixed.reshape((2,) * (N - 2)))
i_j_fixed += 1.0
tmp_i_j_fixed += 1.0
assert torch.allclose(i_j_fixed, tmp_i_j_fixed)

assert torch.allclose(A, B.reshape((2,) * N))
