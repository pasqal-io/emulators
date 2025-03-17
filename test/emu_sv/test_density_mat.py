import math
import torch
from emu_sv.density_matrix_state import DensityMatrix
from emu_sv.state_vector import StateVector

dtype = torch.complex128
device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
gpu = False if device == "cpu" else True
seed = 1337

density_bell_state = (
    1
    / 2
    * torch.tensor(
        [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]],
        dtype=dtype,
        device=device,
    )
)


def test_inner():
    n_atoms = 1
    dummy_mat = torch.zeros((2**n_atoms, 2**n_atoms), dtype=dtype)
    dummy_mat[0, 0] = 1.0
    density_a = DensityMatrix(dummy_mat, gpu=gpu)
    assert density_a.inner(density_a) == 1.0

    dummy_mat = torch.zeros((2**n_atoms, 2**n_atoms), dtype=dtype)
    dummy_mat[1, 1] = 1.0
    density_b = DensityMatrix(dummy_mat, gpu=gpu)

    assert density_a.inner(density_b) == 0.0

    n_atoms = 2

    density_c = DensityMatrix(density_bell_state, gpu=gpu)

    assert density_c.inner(density_c) == 1.0

    dummy_mat = torch.zeros((2**n_atoms, 2**n_atoms), dtype=dtype)
    dummy_mat[0, 0] = 1.0
    density_d = DensityMatrix(dummy_mat, gpu=gpu)

    assert density_c.inner(density_d) == 0.5


def test_from_state_vector():
    n_atoms = 3
    state_vector = StateVector.make(n_atoms, gpu=gpu)

    density = DensityMatrix.from_state_vector(state_vector)
    expected = torch.zeros((2**n_atoms, 2**n_atoms), dtype=dtype, device=device)
    expected[0, 0] = 1.0

    assert torch.allclose(density.matrix, expected)

    bell_state_vec = 1 / math.sqrt(2) * torch.tensor([1, 0, 0, 1], dtype=dtype)
    bell_state = StateVector(bell_state_vec, gpu=gpu)

    density = DensityMatrix.from_state_vector(bell_state)

    assert torch.allclose(density.matrix, density_bell_state)


def test_from_state_string():
    n_atoms = 2
    state_string = {"00": 1.0, "11": 1.0}
    basis = ["0", "1"]
    density = DensityMatrix.from_state_string(
        basis=basis, nqubits=n_atoms, strings=state_string, gpu=gpu
    )

    assert torch.allclose(density.matrix, density_bell_state)

    n_atoms = 3
    state_string = {"111": 1.0}
    basis = ["0", "1"]
    density = DensityMatrix.from_state_string(
        basis=basis, nqubits=n_atoms, strings=state_string, gpu=gpu
    )

    dummu_mat = torch.zeros((2**n_atoms, 2**n_atoms), dtype=dtype, device=device)
    dummu_mat[7, 7] = 1.0
    assert torch.allclose(density.matrix, dummu_mat)


def test_probabilities():
    torch.manual_seed(seed)
    density = DensityMatrix(density_bell_state, gpu=gpu)
    sampling1 = density.sample(1000)

    assert sampling1["11"] == 485
    assert sampling1["00"] == 515

    density = DensityMatrix(torch.eye(8, dtype=dtype), gpu=gpu)
    sampling2 = density.sample(1000)

    assert sampling2["000"] == 144
    assert sampling2["001"] == 105
    assert sampling2["010"] == 130
    assert sampling2["011"] == 122
    assert sampling2["100"] == 103
    assert sampling2["101"] == 128
    assert sampling2["110"] == 150
    assert sampling2["111"] == 118
