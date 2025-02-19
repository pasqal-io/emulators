import torch
import timeit
from emu_sv.hamiltonian import RydbergHamiltonian

torch.manual_seed(0)

dtype = torch.complex128
device = "cpu"


N = 20
omega = torch.randn(N, dtype=dtype, device=device)
delta = torch.randn(N, dtype=dtype, device=device)
interaction_matrix = torch.randn((N, N))
h_custom = RydbergHamiltonian(
    omegas=omega, deltas=delta, interaction_matrix=interaction_matrix, device=device
)
v = torch.randn((2,) * N, dtype=dtype, device=device)


def test():
    h_custom * v


n = 1
elapsed_time = timeit.timeit(test, number=n) / n
print(f"Elapsed time: {elapsed_time:.6f} seconds")

# q = 20. n = 1000 samples
# 0.007771 old
# 0.007934 new apply_x
#
