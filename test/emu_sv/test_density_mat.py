
import torch
from emu_sv.density_matrix_state import DensityMatrix

def test_inner():
    density_a = DensityMatrix(torch.tensor([[1, 0], [0, 0]]))   
    density_b = DensityMatrix(torch.tensor([[1, 0], [0, 0]]))

    assert  density_a.inner(density_b) == 1.0