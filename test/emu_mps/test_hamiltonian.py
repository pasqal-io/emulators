import pytest
from functools import reduce

import torch

from emu_mps.hamiltonian import make_H, update_H
from emu_base.pulser_adapter import HamiltonianType
from emu_base.jump_lindblad_operators import compute_noise_from_lindbladians


#########################################
# Code for building the hamiltonian in
# state vector form. For use in the tests
#########################################


dtype = torch.complex128


def single_gate(i: int, nqubits: int, g: torch.Tensor):
    dim = g.shape[0]
    matrices = [torch.eye(dim, dim, dtype=dtype)] * nqubits
    matrices[i] = g
    return reduce(torch.kron, matrices)


def sigma_x(i: int, nqubits: int, dim: int) -> torch.Tensor:
    σ_x = torch.zeros(dim, dim, dtype=dtype)
    σ_x[0, 1] = 1.0
    σ_x[1, 0] = 1.0
    return single_gate(i, nqubits, σ_x)


def sigma_y(i: int, nqubits: int, dim: int) -> torch.Tensor:
    σ_y = torch.zeros(dim, dim, dtype=dtype)
    σ_y[0, 1] = torch.tensor(-1.0j, dtype=dtype)
    σ_y[1, 0] = torch.tensor(1.0j, dtype=dtype)
    return single_gate(i, nqubits, σ_y)


def pu(i: int, nqubits: int, dim: int):
    n = torch.zeros(dim, dim, dtype=dtype)
    n[1, 1] = 1.0
    return single_gate(i, nqubits, n)


def creation_op(i: int, nqubits: int, dim: int):
    sig_plus = torch.zeros(dim, dim, dtype=dtype)
    sig_plus[0, 1] = 1.0
    return single_gate(i, nqubits, sig_plus)


def n(i: int, j: int, nqubits: int, dim: int):
    n = torch.zeros(dim, dim, dtype=dtype)
    n[1, 1] = 1.0

    matrices = [torch.eye(dim, dim, dtype=dtype)] * nqubits
    matrices[i] = n
    matrices[j] = n
    return reduce(torch.kron, matrices)


def sig_plus_min(i: int, j: int, nqubits: int, dim: int):
    sigma_plus = torch.zeros(dim, dim, dtype=dtype)
    sigma_plus[0, 1] = 1.0
    matrices = [torch.eye(dim, dim, dtype=dtype)] * nqubits
    matrices[i] = sigma_plus  # sigma plus
    matrices[j] = (
        sigma_plus.T
    ).contiguous()  # T results in a non-contiguous, use contiguous
    return reduce(torch.kron, matrices)


def sig_plus_min_xy_rydberg(i: int, j: int, nqubits: int, dim: int):
    sigma_plus = torch.zeros(dim, dim, dtype=dtype)
    sigma_plus[2, 1] = 1.0
    matrices = [torch.eye(dim, dim, dtype=dtype)] * nqubits
    matrices[i] = sigma_plus  # sigma plus
    matrices[j] = (
        sigma_plus.T
    ).contiguous()  # T results in a non-contiguous, use contiguous
    return reduce(torch.kron, matrices)


def sv_hamiltonian(
    inter_matrix: torch.Tensor,
    omega: list[torch.Tensor],
    delta: list[torch.Tensor],
    phi: list[torch.Tensor],
    noise: torch.Tensor,
    dim: int,
    hamiltonian_type: HamiltonianType = HamiltonianType.Rydberg,
) -> torch.Tensor:
    n_qubits = inter_matrix.size(dim=1)
    device = omega[0].device
    h = torch.zeros(dim**n_qubits, dim**n_qubits, dtype=dtype, device=device)
    for i in range(n_qubits):
        h += (
            omega[i]
            * (
                torch.cos(phi[i]) * sigma_x(i, n_qubits, dim)
                + torch.sin(phi[i]) * sigma_y(i, n_qubits, dim)
            ).to(dtype=dtype, device=device)
            / 2
        )
        h -= delta[i] * pu(i, n_qubits, dim).to(dtype=dtype, device=device)
        h += single_gate(i, n_qubits, noise)

        if hamiltonian_type == HamiltonianType.Rydberg:
            for j in range(i + 1, n_qubits):
                h += inter_matrix[i, j] * n(i, j, n_qubits, dim).to(
                    dtype=dtype, device=device
                )
        elif hamiltonian_type == HamiltonianType.XY:
            for j in range(i + 1, n_qubits):
                h += inter_matrix[i, j] * sig_plus_min(i, j, n_qubits, dim).to(
                    dtype=dtype, device=device
                )
                h += inter_matrix[i, j] * sig_plus_min(i, j, n_qubits, dim).T.to(
                    dtype=dtype, device=device
                )

    return h


def sv_hamiltonian_xy_Rydberg(
    inter_matrix: torch.Tensor,
    omega: list[torch.Tensor],
    delta: list[torch.Tensor],
    phi: list[torch.Tensor],
    noise: torch.Tensor,
    dim: int,
    hamiltonian_type: HamiltonianType = HamiltonianType.Rydberg,
    inter_matrix_xy: torch.Tensor = torch.zeros(0, 0),
) -> torch.Tensor:
    n_qubits = inter_matrix.size(dim=1)
    device = omega[0].device
    h = torch.zeros(dim**n_qubits, dim**n_qubits, dtype=dtype, device=device)
    for i in range(n_qubits):
        h += (
            omega[i]
            * (
                torch.cos(phi[i]) * sigma_x(i, n_qubits, dim)
                + torch.sin(phi[i]) * sigma_y(i, n_qubits, dim)
            ).to(dtype=dtype, device=device)
            / 2
        )
        h -= delta[i] * pu(i, n_qubits, dim).to(dtype=dtype, device=device)
        h += single_gate(i, n_qubits, noise)
        if hamiltonian_type == HamiltonianType.RydbergXY:
            for j in range(i + 1, n_qubits):
                h += inter_matrix[i, j] * n(i, j, n_qubits, dim).to(
                    dtype=dtype, device=device
                )
                h += inter_matrix_xy[i, j] * sig_plus_min_xy_rydberg(
                    i, j, n_qubits, dim
                ).to(dtype=dtype, device=device)
                h += inter_matrix_xy[i, j] * sig_plus_min_xy_rydberg(
                    i, j, n_qubits, dim
                ).T.to(dtype=dtype, device=device)
        else:
            raise ValueError("only hamiltonian_xy_Rydberg")

    return h


#########################################


# works for nqubits < 6
# uses the first 10 primes for identifying terms
# when eyeballing the matrices
def create_omega_delta_phi(nqubits: int):
    omega = torch.tensor([2, 3, 4, 5, 7, 11], dtype=dtype)
    delta = torch.tensor([13, 17, 19, 23, 29, 31], dtype=dtype)
    phi = torch.tensor([1.57, 1.58, 1.59, 1.60, 1.61, 1.62])
    return omega[:nqubits], delta[:nqubits], phi[:nqubits]


@pytest.mark.parametrize(
    "basis",
    (("0", "1"), ("g", "r"), ("g", "r", "x"), ("g", "r", "r1")),
    ids=("XY", "Rydberg", "Rydberg-leakage", "RydbergXY"),
)
def test_2_qubit(basis):
    n_atoms = 2
    dim = len(basis)
    if basis == ("g", "r") or ("g", "r", "x"):
        hamiltonian_type = HamiltonianType.Rydberg
    if basis == ("0", "1"):
        hamiltonian_type = HamiltonianType.XY
    if basis == ("g", "r", "r1"):
        hamiltonian_type = HamiltonianType.RydbergXY

    num_gpus = 0
    omega, delta, phi = create_omega_delta_phi(n_atoms)
    if hamiltonian_type == HamiltonianType.Rydberg:
        interaction_matrix = torch.tensor([[0.0000, 5.4202], [5.4202, 0.0000]])
    elif hamiltonian_type == HamiltonianType.XY:
        interaction_matrix = torch.tensor([[0.0000, 3.7000], [3.7000, 0.0000]])
    elif hamiltonian_type == HamiltonianType.RydbergXY:
        interaction_matrix = torch.tensor([[0.0000, 5.4202], [5.4202, 0.0000]])
    if hamiltonian_type != HamiltonianType.RydbergXY:
        ham = make_H(
            interaction_matrix=interaction_matrix,
            hamiltonian_type=hamiltonian_type,
            dim=dim,
            num_gpus_to_use=num_gpus,
        )

    if hamiltonian_type == HamiltonianType.RydbergXY:

        interaction_matrix_xy = torch.tensor([[0.0000, 3.7000], [3.7000, 0.0000]])
        ham = make_H(
            interaction_matrix=interaction_matrix,
            hamiltonian_type=hamiltonian_type,
            dim=dim,
            num_gpus_to_use=num_gpus,
            interaction_matrix_xy=interaction_matrix_xy,
        )

    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
        noise=torch.zeros(dim, dim, dtype=dtype),
    )

    if hamiltonian_type == HamiltonianType.Rydberg:
        assert ham.factors[0].shape == (1, dim, dim, 3)
        assert ham.factors[1].shape == (3, dim, dim, 1)
    elif hamiltonian_type == HamiltonianType.XY:
        assert ham.factors[0].shape == (1, dim, dim, 4)
        assert ham.factors[1].shape == (4, dim, dim, 1)
    elif hamiltonian_type == HamiltonianType.RydbergXY:
        assert ham.factors[0].shape == (1, dim, dim, 5)
        assert ham.factors[1].shape == (5, dim, dim, 1)

    sv = torch.einsum("ijkl,lmno->ijmkno", *(ham.factors)).reshape(
        dim**n_atoms, dim**n_atoms
    )
    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix,
        omega,
        delta,
        phi,
        hamiltonian_type=hamiltonian_type,
        noise=torch.zeros(dim, dim, dtype=dtype),
        dim=dim,
    ).to(dev)

    if hamiltonian_type == HamiltonianType.RydbergXY:
        expected = sv_hamiltonian_xy_Rydberg(
            interaction_matrix,
            omega,
            delta,
            phi,
            hamiltonian_type=hamiltonian_type,
            noise=torch.zeros(dim, dim, dtype=dtype),
            dim=dim,
            inter_matrix_xy=interaction_matrix_xy,
        ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize(
    "basis",
    (("0", "1"), ("g", "r"), ("g", "r", "x"), ("g", "r", "r1")),
    ids=("XY", "Rydberg", "Rydberg-leakage", "RydbergXY"),
)
def test_3_qubit(basis):
    n_atoms = 3
    dim = len(basis)
    if basis == ("g", "r") or ("g", "r", "x"):
        hamiltonian_type = HamiltonianType.Rydberg
    if basis == ("0", "1"):
        hamiltonian_type = HamiltonianType.XY
    if basis == ("g", "r", "r1"):
        hamiltonian_type = HamiltonianType.RydbergXY

    num_gpus = 0
    omega, delta, phi = create_omega_delta_phi(n_atoms)
    interaction_matrix = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    interaction_matrix.fill_diagonal_(0)

    if hamiltonian_type != HamiltonianType.RydbergXY:
        ham = make_H(
            interaction_matrix=interaction_matrix,
            hamiltonian_type=hamiltonian_type,
            dim=dim,
            num_gpus_to_use=num_gpus,
        )

    if hamiltonian_type == HamiltonianType.RydbergXY:

        interaction_matrix_xy = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
        interaction_matrix_xy = (interaction_matrix_xy + interaction_matrix_xy.T) / 2
        interaction_matrix_xy.fill_diagonal_(0)
        ham = make_H(
            interaction_matrix=interaction_matrix,
            hamiltonian_type=hamiltonian_type,
            dim=dim,
            num_gpus_to_use=num_gpus,
            interaction_matrix_xy=interaction_matrix_xy,
        )

    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
        noise=torch.zeros(dim, dim, dtype=dtype),
    )

    if hamiltonian_type == HamiltonianType.Rydberg:
        assert ham.factors[0].shape == (1, dim, dim, 3)
        assert ham.factors[1].shape == (3, dim, dim, 3)
        assert ham.factors[2].shape == (3, dim, dim, 1)
    elif hamiltonian_type == HamiltonianType.XY:
        assert ham.factors[0].shape == (1, dim, dim, 4)
        assert ham.factors[1].shape == (4, dim, dim, 4)
        assert ham.factors[2].shape == (4, dim, dim, 1)
    elif hamiltonian_type == HamiltonianType.RydbergXY:
        assert ham.factors[0].shape == (1, dim, dim, 5)
        assert ham.factors[1].shape == (5, dim, dim, 5)
        assert ham.factors[2].shape == (5, dim, dim, 1)

    # sv = torch.einsum("ijkl,lmno->ijmkno", *(ham.factors)).reshape(
    sv = torch.einsum("ijkl,lmno,opqr->ijmpknqr", *(ham.factors)).reshape(
        dim**n_atoms, dim**n_atoms
    )
    dev = sv.device  # could be cpu or gpu depending on Config
    if hamiltonian_type != HamiltonianType.RydbergXY:
        expected = sv_hamiltonian(
            interaction_matrix,
            omega,
            delta,
            phi,
            hamiltonian_type=hamiltonian_type,
            noise=torch.zeros(dim, dim, dtype=dtype),
            dim=dim,
        ).to(dev)

    if hamiltonian_type == HamiltonianType.RydbergXY:
        expected = sv_hamiltonian_xy_Rydberg(
            interaction_matrix,
            omega,
            delta,
            phi,
            hamiltonian_type=hamiltonian_type,
            noise=torch.zeros(dim, dim, dtype=dtype),
            dim=dim,
            inter_matrix_xy=interaction_matrix_xy,
        ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r"), ("g", "r", "x")))
def test_noise(basis):
    n_atoms = 2
    dim = len(basis)
    if basis == ("g", "r") or ("g", "r", "x"):
        hamiltonian_type = HamiltonianType.Rydberg

    elif basis == ("0", "1"):
        hamiltonian_type = HamiltonianType.XY

    num_gpus = 0
    omega = torch.tensor([0.0, 0.0], dtype=dtype)
    delta = torch.tensor([0.0, 0.0], dtype=dtype)
    phi = torch.tensor([0.0, 0.0], dtype=dtype)

    # Interaction term [0,1] = 0.0 to erase the interaction
    interaction_matrix = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

    rate = 0.234
    # noise = torch.zeros(dim,dim,dtype=dtype)
    if dim == 2:
        noise = -1j / 2.0 * torch.tensor([[0, 0], [0, rate]], dtype=dtype)
    elif dim == 3:
        noise = (
            -1j / 2.0 * torch.tensor([[0, 0, 0], [0, rate, 0], [0, 0, 0]], dtype=dtype)
        )

    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=hamiltonian_type,
        dim=dim,
        num_gpus_to_use=num_gpus,
    )

    update_H(hamiltonian=ham, omega=omega, delta=delta, phi=phi, noise=noise)

    sv = torch.einsum("ijkl,lmno->ijmkno", *(ham.factors)).reshape(
        dim**n_atoms, dim**n_atoms
    )
    expected = sv_hamiltonian(
        interaction_matrix,
        omega,
        delta,
        phi,
        noise=noise,
        hamiltonian_type=hamiltonian_type,
        dim=dim,
    ).to(sv.device)

    assert abs(sv[1, 1] - (-1j / 2.0 * rate)) < 1e-10

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize(
    "basis",
    (("0", "1"), ("g", "r"), ("g", "r", "x"), ("g", "r", "r1")),
    ids=("XY", "Rydberg", "Rydberg-leakage", "RydbergXY"),
)
def test_4_qubit(basis):
    n_atoms = 4
    dim = len(basis)
    interaction_matrix_xy = torch.zeros(0, 0)
    if basis == ("g", "r") or ("g", "r", "x"):
        hamiltonian_type = HamiltonianType.Rydberg
    if basis == ("0", "1"):
        hamiltonian_type = HamiltonianType.XY
    if basis == ("g", "r", "r1"):  #
        hamiltonian_type = HamiltonianType.RydbergXY
        interaction_matrix_xy = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
        interaction_matrix_xy = (interaction_matrix_xy + interaction_matrix_xy.T) / 2
        interaction_matrix_xy.fill_diagonal_(0)

    num_gpus = 0
    omega, delta, phi = create_omega_delta_phi(n_atoms)

    interaction_matrix = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    interaction_matrix.fill_diagonal_(0)
    if hamiltonian_type != HamiltonianType.RydbergXY:
        ham = make_H(
            interaction_matrix=interaction_matrix,
            hamiltonian_type=hamiltonian_type,
            dim=dim,
            num_gpus_to_use=num_gpus,
        )
    if hamiltonian_type == HamiltonianType.RydbergXY:
        ham = make_H(
            interaction_matrix=interaction_matrix,
            hamiltonian_type=hamiltonian_type,
            dim=dim,
            num_gpus_to_use=num_gpus,
            interaction_matrix_xy=interaction_matrix_xy,
        )
    noise = torch.zeros(dim, dim, dtype=dtype)
    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
        noise=noise,
    )
    if hamiltonian_type == HamiltonianType.Rydberg:
        assert ham.factors[0].shape == (1, dim, dim, 3)
        assert ham.factors[1].shape == (3, dim, dim, 4)
        assert ham.factors[2].shape == (4, dim, dim, 3)
        assert ham.factors[3].shape == (3, dim, dim, 1)
    elif hamiltonian_type == HamiltonianType.XY:
        assert ham.factors[0].shape == (1, dim, dim, 4)
        assert ham.factors[1].shape == (4, dim, dim, 6)
        assert ham.factors[2].shape == (6, dim, dim, 4)
        assert ham.factors[3].shape == (4, dim, dim, 1)
    elif hamiltonian_type == HamiltonianType.RydbergXY:
        assert ham.factors[0].shape == (1, dim, dim, 5)
        assert ham.factors[1].shape == (5, dim, dim, 8)
        assert ham.factors[2].shape == (8, dim, dim, 5)
        assert ham.factors[3].shape == (5, dim, dim, 1)

    sv = torch.einsum("ijkl,lmno,opqr,rstu->ijmpsknqtu", *(ham.factors)).reshape(
        dim**n_atoms, dim**n_atoms
    )
    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix,
        omega,
        delta,
        phi,
        dim=dim,
        noise=noise,
        hamiltonian_type=hamiltonian_type,
    ).to(dev)

    if hamiltonian_type == HamiltonianType.RydbergXY:
        expected = sv_hamiltonian_xy_Rydberg(
            interaction_matrix,
            omega,
            delta,
            phi,
            dim=dim,
            noise=noise,
            hamiltonian_type=hamiltonian_type,
            inter_matrix_xy=interaction_matrix_xy,
        ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize(
    "basis",
    (("0", "1"), ("g", "r"), ("g", "r", "x"), ("g", "r", "r1")),
    ids=("XY", "Rydberg", "Rydberg-leakage", "RydbergXY"),
)
def test_5_qubit(basis):
    n_atoms = 5
    dim = len(basis)
    if basis == ("g", "r") or ("g", "r", "x"):
        hamiltonian_type = HamiltonianType.Rydberg
    if basis == ("0", "1"):
        hamiltonian_type = HamiltonianType.XY
    if basis == ("g", "r", "r1"):
        hamiltonian_type = HamiltonianType.RydbergXY
        interaction_matrix_xy = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
        interaction_matrix_xy = (interaction_matrix_xy + interaction_matrix_xy.T) / 2
        interaction_matrix_xy.fill_diagonal_(0)

    num_gpus = 0
    omega, delta, phi = create_omega_delta_phi(n_atoms)

    interaction_matrix = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    interaction_matrix.fill_diagonal_(0)
    if hamiltonian_type != HamiltonianType.RydbergXY:
        ham = make_H(
            interaction_matrix=interaction_matrix,
            hamiltonian_type=hamiltonian_type,
            dim=dim,
            num_gpus_to_use=num_gpus,
        )

    if hamiltonian_type == HamiltonianType.RydbergXY:
        ham = make_H(
            interaction_matrix=interaction_matrix,
            hamiltonian_type=hamiltonian_type,
            dim=dim,
            num_gpus_to_use=num_gpus,
            interaction_matrix_xy=interaction_matrix_xy,
        )

    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
        noise=torch.zeros(dim, dim, dtype=dtype),
    )

    if hamiltonian_type == HamiltonianType.Rydberg:
        assert ham.factors[0].shape == (1, dim, dim, 3)
        assert ham.factors[1].shape == (3, dim, dim, 4)
        assert ham.factors[2].shape == (4, dim, dim, 4)
        assert ham.factors[3].shape == (4, dim, dim, 3)
        assert ham.factors[4].shape == (3, dim, dim, 1)
    elif hamiltonian_type == HamiltonianType.XY:
        assert ham.factors[0].shape == (1, dim, dim, 4)
        assert ham.factors[1].shape == (4, dim, dim, 6)
        assert ham.factors[2].shape == (6, dim, dim, 6)
        assert ham.factors[3].shape == (6, dim, dim, 4)
        assert ham.factors[4].shape == (4, dim, dim, 1)
    elif hamiltonian_type == HamiltonianType.RydbergXY:
        assert ham.factors[0].shape == (1, dim, dim, 5)
        assert ham.factors[1].shape == (5, dim, dim, 8)
        assert ham.factors[2].shape == (8, dim, dim, 8)
        assert ham.factors[3].shape == (8, dim, dim, 5)
        assert ham.factors[4].shape == (5, dim, dim, 1)

    sv = torch.einsum("abcd,defg,ghij,jklm,mnop->abehkncfilop", *(ham.factors)).reshape(
        dim**n_atoms, dim**n_atoms
    )
    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix,
        omega,
        delta,
        phi,
        hamiltonian_type=hamiltonian_type,
        dim=dim,
        noise=torch.zeros(dim, dim, dtype=dtype),
    ).to(dev)

    if hamiltonian_type == HamiltonianType.RydbergXY:
        expected = sv_hamiltonian_xy_Rydberg(
            interaction_matrix,
            omega,
            delta,
            phi,
            hamiltonian_type=hamiltonian_type,
            dim=dim,
            noise=torch.zeros(dim, dim, dtype=dtype),
            inter_matrix_xy=interaction_matrix_xy,
        ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r")))
def test_9_qubit_noise(basis):
    """Hall of fame: this test caught a bug in the original implementation."""
    n_atoms = 9
    dim = len(basis)
    if basis == ("g", "r"):
        hamiltonian_type = HamiltonianType.Rydberg

    elif basis == ("0", "1"):
        hamiltonian_type = HamiltonianType.XY

    num_gpus = 0
    omega = torch.tensor([12.566370614359172] * n_atoms, dtype=dtype)
    delta = torch.tensor([10.771174812307862] * n_atoms, dtype=dtype)
    phi = torch.tensor([torch.pi] * n_atoms, dtype=dtype)

    interaction_matrix = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    interaction_matrix.fill_diagonal_(0)

    lindbladians = [
        torch.tensor([[-5.0, 4.0], [2.0, 5.0]], dtype=dtype),
        torch.tensor([[2.0, 3.0], [1.5, 5.0j]], dtype=dtype),
        torch.tensor([[-2.5j + 0.5, 2.3], [1.0, 2.0]], dtype=dtype),
    ]

    noise = compute_noise_from_lindbladians(lindbladians, dim=dim)

    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=hamiltonian_type,
        dim=dim,
        num_gpus_to_use=num_gpus,
    )
    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
        noise=noise,
    )

    sv = torch.einsum(
        "abcd,defg,ghij,jklm,mnop,pqrs,stuv,vwxy,yzAB->abehknqtwzcfiloruxAB",
        *(ham.factors),
    ).reshape(dim**n_atoms, dim**n_atoms)

    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix,
        omega,
        delta,
        phi,
        noise,
        dim=dim,
        hamiltonian_type=hamiltonian_type,
    ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize("basis", (("g", "r", "r1"),))
def test_6_qubit_(basis):
    """Hall of fame: this test caught a bug in the original implementation."""
    n_atoms = 6
    dim = len(basis)

    hamiltonian_type = HamiltonianType.RydbergXY

    num_gpus = 0
    omega = torch.tensor([12.566370614359172] * n_atoms, dtype=dtype)
    delta = torch.tensor([10.771174812307862] * n_atoms, dtype=dtype)
    phi = torch.tensor([torch.pi] * n_atoms, dtype=dtype)

    interaction_matrix = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    interaction_matrix.fill_diagonal_(0)

    interaction_matrix_xy = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
    interaction_matrix_xy = (interaction_matrix_xy + interaction_matrix_xy.T) / 2
    interaction_matrix_xy.fill_diagonal_(0)

    # lindbladians = [
    # torch.tensor([[-5.0, 4.0], [2.0, 5.0]], dtype=dtype),
    # torch.tensor([[2.0, 3.0], [1.5, 5.0j]], dtype=dtype),
    # torch.tensor([[-2.5j + 0.5, 2.3], [1.0, 2.0]], dtype=dtype),
    # ]

    # noise = compute_noise_from_lindbladians(lindbladians, dim=dim)
    noise = torch.zeros(dim, dim, dtype=dtype)
    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=hamiltonian_type,
        dim=dim,
        num_gpus_to_use=num_gpus,
        interaction_matrix_xy=interaction_matrix_xy,
    )
    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
        noise=torch.zeros(dim, dim, dtype=dtype),
    )

    sv = torch.einsum(
        # "abcd,defg,ghij,jklm,mnop,pqrs,stuv,vwxy,yzAB->abehknqtwzcfiloruxAB",
        "abcd,defg,ghij,jklm,mnop,pqrs->abehknqcfilors",
        *(ham.factors),
    ).reshape(dim**n_atoms, dim**n_atoms)

    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian_xy_Rydberg(
        interaction_matrix,
        omega,
        delta,
        phi,
        noise,
        dim=dim,
        hamiltonian_type=hamiltonian_type,
        inter_matrix_xy=interaction_matrix_xy,
    ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize("basis", (("g", "r", "r1"),))
def test_7_qubit_(basis):
    """Hall of fame: this test caught a bug in the original implementation."""
    n_atoms = 7
    dim = len(basis)

    hamiltonian_type = HamiltonianType.RydbergXY

    num_gpus = 0
    omega = torch.tensor([12.566370614359172] * n_atoms, dtype=dtype)
    delta = torch.tensor([10.771174812307862] * n_atoms, dtype=dtype)
    phi = torch.tensor([torch.pi] * n_atoms, dtype=dtype)

    interaction_matrix = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    interaction_matrix.fill_diagonal_(0)

    interaction_matrix_xy = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
    interaction_matrix_xy = (interaction_matrix_xy + interaction_matrix_xy.T) / 2
    interaction_matrix_xy.fill_diagonal_(0)

    noise = torch.zeros(dim, dim, dtype=dtype)
    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=hamiltonian_type,
        dim=dim,
        num_gpus_to_use=num_gpus,
        interaction_matrix_xy=interaction_matrix_xy,
    )
    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
        noise=torch.zeros(dim, dim, dtype=dtype),
    )

    sv = torch.einsum(
        # "abcd,defg,ghij,jklm,mnop,pqrs,stuv,vwxy,yzAB->abehknqtwzcfiloruxAB",
        # "abcd,defg,ghij,jklm,mnop,pqrs->abehknqcfilors",
        "abcd,defg,ghij,jklm,mnop,pqrs,stuv->abehknqtcfiloruv",
        *(ham.factors),
    ).reshape(dim**n_atoms, dim**n_atoms)

    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian_xy_Rydberg(
        interaction_matrix,
        omega,
        delta,
        phi,
        noise,
        dim=dim,
        hamiltonian_type=hamiltonian_type,
        inter_matrix_xy=interaction_matrix_xy,
    ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize("basis", (("g", "r", "r1"),))
def test_8_qubit_(basis):
    """Hall of fame: this test caught a bug in the original implementation."""
    n_atoms = 8
    dim = len(basis)

    hamiltonian_type = HamiltonianType.RydbergXY

    num_gpus = 0
    omega = torch.tensor([12.566370614359172] * n_atoms, dtype=dtype)
    delta = torch.tensor([10.771174812307862] * n_atoms, dtype=dtype)
    phi = torch.tensor([torch.pi] * n_atoms, dtype=dtype)

    interaction_matrix = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    interaction_matrix.fill_diagonal_(0)

    interaction_matrix_xy = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
    interaction_matrix_xy = (interaction_matrix_xy + interaction_matrix_xy.T) / 2
    interaction_matrix_xy.fill_diagonal_(0)

    noise = torch.zeros(dim, dim, dtype=dtype)
    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=hamiltonian_type,
        dim=dim,
        num_gpus_to_use=num_gpus,
        interaction_matrix_xy=interaction_matrix_xy,
    )
    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
        noise=torch.zeros(dim, dim, dtype=dtype),
    )

    sv = torch.einsum(
        # "abcd,defg,ghij,jklm,mnop,pqrs,stuv,vwxy,yzAB->abehknqtwzcfiloruxAB",
        # "abcd,defg,ghij,jklm,mnop,pqrs->abehknqcfilors",
        # "abcd,defg,ghij,jklm,mnop,pqrs,stuv->abehknqtcfiloruv",
        "abcd,defg,ghij,jklm,mnop,pqrs,stuv,vwxy->abehknqtwcfiloruxy",
        *(ham.factors),
    ).reshape(dim**n_atoms, dim**n_atoms)

    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian_xy_Rydberg(
        interaction_matrix,
        omega,
        delta,
        phi,
        noise,
        dim=dim,
        hamiltonian_type=hamiltonian_type,
        inter_matrix_xy=interaction_matrix_xy,
    ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize("basis", (("0", "1", "x"), ("g", "r", "x"), ("g", "r", "r1")))
def test_6_qubit_3_level_noise(basis):
    n_atoms = 6
    dim = len(basis)
    if basis == ("g", "r", "x"):
        hamiltonian_type = HamiltonianType.Rydberg
    elif basis == ("0", "1", "x"):
        hamiltonian_type = HamiltonianType.XY
    elif basis == ("g", "r", "r1"):
        hamiltonian_type = HamiltonianType.RydbergXY

    num_gpus = 0
    omega = torch.tensor([12.566370614359172] * n_atoms, dtype=dtype)
    delta = torch.tensor([10.771174812307862] * n_atoms, dtype=dtype)
    phi = torch.tensor([torch.pi] * n_atoms, dtype=dtype)

    interaction_matrix = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    interaction_matrix.fill_diagonal_(0)

    lindbladians = [
        torch.tensor([[-5.0, 4.0, 0.0], [2.0, 5.0, 1.0], [0.0, 0.0, 2.0]], dtype=dtype),
        torch.tensor([[2.0, 3.0, 3.0], [1.5, 5.0j, 4.0], [0.0, 0.0, 5.0]], dtype=dtype),
        torch.tensor(
            [[-2.5j + 0.5, 2.3, 7.0], [1.0, 2.0, 9.0], [0.0, 0.0, 11.0]], dtype=dtype
        ),
    ]

    noise = compute_noise_from_lindbladians(lindbladians, dim=dim)

    if hamiltonian_type != HamiltonianType.RydbergXY:
        ham = make_H(
            interaction_matrix=interaction_matrix,
            hamiltonian_type=hamiltonian_type,
            dim=dim,
            num_gpus_to_use=num_gpus,
        )

    if hamiltonian_type == HamiltonianType.RydbergXY:
        interaction_matrix_xy = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
        interaction_matrix_xy = (interaction_matrix_xy + interaction_matrix_xy.T) / 2
        interaction_matrix_xy.fill_diagonal_(0)
        ham = make_H(
            interaction_matrix=interaction_matrix,
            hamiltonian_type=hamiltonian_type,
            dim=dim,
            num_gpus_to_use=num_gpus,
            interaction_matrix_xy=interaction_matrix_xy,
        )

    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
        noise=noise,
    )

    sv = torch.einsum(
        "abcd,defg,ghij,jklm,mnop,pqrs->abehknqcfilors",
        *(ham.factors),
    ).reshape(dim**n_atoms, dim**n_atoms)

    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix,
        omega,
        delta,
        phi,
        noise,
        dim=dim,
        hamiltonian_type=hamiltonian_type,
    ).to(dev)
    if hamiltonian_type == HamiltonianType.RydbergXY:
        expected = sv_hamiltonian_xy_Rydberg(
            interaction_matrix,
            omega,
            delta,
            phi,
            noise,
            dim=dim,
            hamiltonian_type=hamiltonian_type,
            inter_matrix_xy=interaction_matrix_xy,
        ).to(dev)
    print(sv)
    print(expected)
    assert torch.allclose(sv, expected)


def test_differentiation():
    """Basic test for torch grad."""
    # The interaction term of the Hamiltonian will not affect the differentiation.
    # Thus, we are not testing with XY interaction
    n = 5
    num_gpus = 0

    omega = torch.tensor([1.0] * n, dtype=dtype, requires_grad=True)
    delta = torch.tensor([1.0] * n, dtype=dtype, requires_grad=True)
    phi = torch.tensor([torch.pi] * n, dtype=dtype, requires_grad=True)

    interaction_matrix = torch.tensor(
        [
            [0.0000, 5.4202, 5.4202, 0.6775, 43.3613],
            [0.0000, 0.0000, 0.6775, 5.4202, 43.3613],
            [0.0000, 0.0000, 0.0000, 5.4202, 43.3613],
            [0.0000, 0.0000, 0.0000, 0.0000, 43.3613],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ]
    )
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2

    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=HamiltonianType.Rydberg,
        num_gpus_to_use=num_gpus,
    )
    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
        noise=torch.zeros(2, 2, dtype=dtype),
    )

    sv = torch.einsum("abcd,defg,ghij,jklm,mnop->abehkncfilop", *(ham.factors)).reshape(
        1 << n, 1 << n
    )

    # loop over each element in the state-vector form of the hamiltonian,
    # and assert that it depends
    # on the omegas and deltas in the correct way.
    for i in range(32):
        for j in range(32):
            expected_delta_diff = (
                torch.zeros(5, dtype=dtype)
                if i != j
                else torch.tensor(
                    [-1.0 if (i & (1 << (n - k - 1)) != 0) else 0.0 for k in range(n)],
                    dtype=dtype,
                )
            )
            assert torch.allclose(
                torch.autograd.grad(sv[i, j].real, delta, retain_graph=True)[0],
                expected_delta_diff,
            )

            expected_omega_diff = torch.zeros(n, dtype=dtype)
            if (i ^ j).bit_count() == 1:
                expected_omega_diff[n - (i ^ j).bit_length()] = -0.5

            assert torch.allclose(
                torch.autograd.grad(sv[i, j].real, omega, retain_graph=True)[0],
                expected_omega_diff,
            )
            expected_phi_diff = torch.zeros(n, dtype=dtype)
            if (i ^ j).bit_count() == 1:
                expected_phi_diff[n - (i ^ j).bit_length()] = 0.5

            assert torch.allclose(
                abs(torch.autograd.grad(sv[i, j].real, phi, retain_graph=True)[0]).to(
                    dtype=dtype
                ),
                expected_phi_diff,
            )


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r")))
def test_truncation_random(basis):
    n_atoms = 9
    dim = len(basis)
    if basis == ("g", "r"):
        hamiltonian_type = HamiltonianType.Rydberg

    elif basis == ("0", "1"):
        hamiltonian_type = HamiltonianType.XY

    num_gpus = 0
    omega = torch.tensor([12.566370614359172] * n_atoms, dtype=dtype)
    delta = torch.zeros(n_atoms, dtype=dtype)
    phi = torch.zeros(n_atoms, dtype=dtype)

    interaction_matrix = torch.randn(n_atoms, n_atoms, dtype=torch.float64)
    interaction_matrix = 0.5 * (interaction_matrix + interaction_matrix.T)
    interaction_matrix.fill_diagonal_(0)

    interaction_matrix[interaction_matrix < 0.7] = 0.0
    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=hamiltonian_type,
        dim=dim,
        num_gpus_to_use=num_gpus,
    )
    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
        noise=torch.zeros(dim, dim, dtype=dtype),
    )

    sv = torch.einsum(
        "abcd,defg,ghij,jklm,mnop,pqrs,stuv,vwxy,yzAB->abehknqtwzcfiloruxAB",
        *(ham.factors),
    ).reshape(dim**n_atoms, dim**n_atoms)

    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix,
        omega,
        delta,
        phi,
        dim=dim,
        noise=torch.zeros(dim, dim),
        hamiltonian_type=hamiltonian_type,
    ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r"), ("g", "r", "x")))
def test_truncation_nn(basis):
    n_atoms = 5
    dim = len(basis)
    if basis == ("g", "r") or ("g", "r", "x"):
        hamiltonian_type = HamiltonianType.Rydberg

    elif basis == ("0", "1"):
        hamiltonian_type = HamiltonianType.XY

    omega = torch.zeros(n_atoms, dtype=dtype)
    delta = torch.zeros(n_atoms, dtype=dtype)
    phi = torch.zeros(n_atoms, dtype=dtype)

    interaction_matrix = torch.diag(
        torch.tensor([1.0] * (n_atoms - 1), dtype=torch.float64), 1
    )
    interaction_matrix = interaction_matrix + interaction_matrix.T
    ham = make_H(
        interaction_matrix=interaction_matrix,
        num_gpus_to_use=0,
        hamiltonian_type=hamiltonian_type,
        dim=dim,
    )

    sv = torch.einsum("abcd,defg,ghij,jklm,mnop->abehkncfilop", *(ham.factors)).reshape(
        dim**n_atoms, dim**n_atoms
    )

    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix,
        omega,
        delta,
        phi,
        torch.zeros(dim, dim),
        dim=dim,
        hamiltonian_type=hamiltonian_type,
    ).to(dev)

    if hamiltonian_type == HamiltonianType.Rydberg:
        size = 3
    elif hamiltonian_type == HamiltonianType.XY:
        size = 4
    else:
        raise NotImplementedError("Extend the tests")

    assert ham.factors[0].shape == (1, dim, dim, size)
    assert ham.factors[1].shape == (size, dim, dim, size)
    assert ham.factors[2].shape == (size, dim, dim, size)
    assert ham.factors[3].shape == (size, dim, dim, size)
    assert ham.factors[4].shape == (size, dim, dim, 1)

    assert torch.allclose(
        sv,
        expected,
    )
