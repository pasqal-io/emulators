import math
import torch
from emu_sv.density_matrix_state import DensityMatrix
from emu_sv.state_vector import StateVector
from test.utils_testing.utils_testing import random_density_matrix
from pytest import approx

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

density_bell_state_complex = (
    1
    / 2
    * torch.tensor(
        [[1, 0, 0, -1j], [0, 0, 0, 0], [0, 0, 0, 0], [1j, 0, 0, 1]],
        dtype=dtype,
        device=device,
    )
)


def test_inner():
    torch.manual_seed(seed)

    n_atoms = 1

    dummy_mat = random_density_matrix(n_atoms)
    dense_matrix = DensityMatrix(dummy_mat, gpu=gpu)
    a = dummy_mat[0, 0].item()
    c = dummy_mat[0, 1].item()
    # for A = [[a,c],[c^†,b]], b=1-a
    assert dense_matrix.inner(dense_matrix) == approx(
        1 - 2 * a + 2 * a * a.conjugate() + 2 * c * c.conjugate()
    )

    n_atoms = 2

    density_c = DensityMatrix(density_bell_state, gpu=gpu)

    assert density_c.inner(density_c) == 1.0

    dummy_mat = torch.zeros((2**n_atoms, 2**n_atoms), dtype=dtype)
    dummy_mat[0, 0] = 1.0
    density_d = DensityMatrix(dummy_mat, gpu=gpu)

    assert density_c.inner(density_d) == 0.5

    n_atoms = 5
    dummy_mat1 = random_density_matrix(n_atoms)
    dense_matrix = DensityMatrix(dummy_mat1, gpu=gpu)

    base_mat = torch.zeros((2**n_atoms, 2**n_atoms), dtype=dtype)
    base_mat[n_atoms - 1, n_atoms - 1] = 1.0
    base_state = DensityMatrix(base_mat, gpu=gpu)
    # getting the [24,24] element of the density matrix
    assert dense_matrix.inner(base_state) == approx(0.035517495)


def test_make():
    n_atoms = 2
    ground_density = DensityMatrix.make(n_atoms, gpu=gpu)
    zeros_mat = torch.zeros(2**n_atoms, 2**n_atoms, dtype=dtype, device=device)
    zeros_mat[0, 0] = 1.0
    assert torch.allclose(ground_density.matrix, zeros_mat)

    n_atoms = 3
    ground_density = DensityMatrix.make(n_atoms, gpu=gpu)
    zeros_mat = torch.zeros(2**n_atoms, 2**n_atoms, dtype=dtype, device=device)
    zeros_mat[0, 0] = 1.0
    assert torch.allclose(ground_density.matrix, zeros_mat)


def test_from_state_vector():
    bell_state_vec = 1 / math.sqrt(2) * torch.tensor([1, 0, 0, 1], dtype=dtype)
    bell_state = StateVector(bell_state_vec, gpu=gpu)

    density = DensityMatrix.from_state_vector(bell_state)

    assert torch.allclose(density.matrix, density_bell_state)

    n_atoms = 6
    state_vector = StateVector.make(n_atoms, gpu=gpu)

    density = DensityMatrix.from_state_vector(state_vector)
    expected = torch.zeros((2**n_atoms, 2**n_atoms), dtype=dtype, device=device)
    expected[0, 0] = 1.0

    assert torch.allclose(density.matrix, expected)


def test_from_state_string():
    n_atoms = 2
    state_string = {"00": 1.0, "11": 1.0j}
    basis = ["0", "1"]
    density = DensityMatrix.from_state_string(
        basis=basis, nqubits=n_atoms, strings=state_string, gpu=gpu
    )

    assert torch.allclose(density.matrix, density_bell_state_complex)

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

    # testing a random matrix
    n_atoms = 8
    dummy_mat = random_density_matrix(n_atoms)
    density_mat = DensityMatrix(dummy_mat, gpu=gpu)

    sampling3 = density_mat.sample(3000)

    assert sampling3["0" * n_atoms] == 11
    assert sampling3["1" * n_atoms] == 13
    assert sampling3["01010101"] == 14
    assert sampling3["10101001"] == 11
