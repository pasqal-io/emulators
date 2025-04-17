import pytest
from functools import reduce

import torch

from emu_mps.hamiltonian import make_H, update_H
from emu_base.pulser_adapter import HamiltonianType
from emu_mps.noise import compute_noise_from_lindbladians


#########################################
# Code for building the hamiltonian in
# state vector form. For use in the tests
#########################################


dtype = torch.complex128


def single_gate(i: int, nqubits: int, g: torch.Tensor):
    matrices = [torch.eye(2, 2, dtype=dtype)] * nqubits
    matrices[i] = g
    return reduce(torch.kron, matrices)


def sigma_x(i: int, nqubits: int) -> torch.Tensor:
    σ_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    return single_gate(i, nqubits, σ_x)


def sigma_y(i: int, nqubits: int) -> torch.Tensor:
    σ_y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype)
    return single_gate(i, nqubits, σ_y)


def pu(i, nqubits):
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
    return single_gate(i, nqubits, n)


def creation_op(i, nqubits):
    sig_plus = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    return single_gate(i, nqubits, sig_plus)


def n(i, j, nqubits):
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
    matrices = [torch.eye(2, 2, dtype=dtype)] * nqubits
    matrices[i] = n
    matrices[j] = n
    return reduce(torch.kron, matrices)


def sig_plus_min(i, j, nqubits):
    sigma_plus = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    matrices = [torch.eye(2, 2, dtype=dtype)] * nqubits
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
    noise: torch.Tensor = torch.zeros(2, 2),
    hamiltonian_type: HamiltonianType = HamiltonianType.Rydberg,
) -> torch.Tensor:
    n_qubits = inter_matrix.size(dim=1)
    device = omega[0].device
    h = torch.zeros(2**n_qubits, 2**n_qubits, dtype=dtype, device=device)
    for i in range(n_qubits):
        h += (
            omega[i]
            * (
                torch.cos(phi[i]) * sigma_x(i, n_qubits)
                + torch.sin(phi[i]) * sigma_y(i, n_qubits)
            ).to(dtype=dtype, device=device)
            / 2
        )
        h -= delta[i] * pu(i, n_qubits).to(dtype=dtype, device=device)
        h += single_gate(i, n_qubits, noise)

        if hamiltonian_type == HamiltonianType.Rydberg:
            for j in range(i + 1, n_qubits):
                h += inter_matrix[i, j] * n(i, j, n_qubits).to(dtype=dtype, device=device)
        elif hamiltonian_type == HamiltonianType.XY:
            for j in range(i + 1, n_qubits):
                h += inter_matrix[i, j] * sig_plus_min(i, j, n_qubits).to(
                    dtype=dtype, device=device
                )
                h += inter_matrix[i, j] * sig_plus_min(i, j, n_qubits).T.to(
                    dtype=dtype, device=device
                )
        else:
            raise ValueError(f"unsupported interaction type {hamiltonian_type}")

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
    "hamiltonian_type",
    [HamiltonianType.Rydberg, HamiltonianType.XY],
)
def test_2_qubit(hamiltonian_type):
    num_gpus = 0
    omega, delta, phi = create_omega_delta_phi(2)
    if hamiltonian_type == HamiltonianType.Rydberg:
        interaction_matrix = torch.tensor([[0.0000, 5.4202], [5.4202, 0.0000]])
    elif hamiltonian_type == HamiltonianType.XY:
        interaction_matrix = torch.tensor([[0.0000, 3.7000], [3.7000, 0.0000]])

    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=hamiltonian_type,
        num_gpus_to_use=num_gpus,
    )

    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
    )

    if hamiltonian_type == HamiltonianType.Rydberg:
        assert ham.factors[0].shape == (1, 2, 2, 3)
        assert ham.factors[1].shape == (3, 2, 2, 1)
    elif hamiltonian_type == HamiltonianType.XY:
        assert ham.factors[0].shape == (1, 2, 2, 4)
        assert ham.factors[1].shape == (4, 2, 2, 1)

    sv = torch.einsum("ijkl,lmno->ijmkno", *(ham.factors)).reshape(4, 4)
    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix, omega, delta, phi, hamiltonian_type=hamiltonian_type
    ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize(
    "hamiltonian_type",
    [
        HamiltonianType.Rydberg,
        HamiltonianType.XY,
    ],
)
def test_noise(hamiltonian_type):
    num_gpus = 0
    omega = torch.tensor([0.0, 0.0], dtype=dtype)
    delta = torch.tensor([0.0, 0.0], dtype=dtype)
    phi = torch.tensor([0.0, 0.0], dtype=dtype)

    # Interaction term [0,1] = 0.0 to erase the interaction
    interaction_matrix = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

    rate = 0.234
    noise = -1j / 2.0 * torch.tensor([[0, 0], [0, rate]], dtype=dtype)

    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=hamiltonian_type,
        num_gpus_to_use=num_gpus,
    )

    update_H(hamiltonian=ham, omega=omega, delta=delta, phi=phi, noise=noise)

    sv = torch.einsum("ijkl,lmno->ijmkno", *(ham.factors)).reshape(4, 4)
    expected = sv_hamiltonian(
        interaction_matrix,
        omega,
        delta,
        phi,
        noise=noise,
        hamiltonian_type=hamiltonian_type,
    ).to(sv.device)

    assert abs(sv[1, 1] - (-1j / 2.0 * rate)) < 1e-10

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize(
    "hamiltonian_type",
    [
        HamiltonianType.Rydberg,
        HamiltonianType.XY,
    ],
)
def test_4_qubit(hamiltonian_type):
    num_gpus = 0
    omega, delta, phi = create_omega_delta_phi(4)

    interaction_matrix = torch.randn(4, 4, dtype=torch.float64)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    interaction_matrix.fill_diagonal_(0)

    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=hamiltonian_type,
        num_gpus_to_use=num_gpus,
    )
    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
    )
    if hamiltonian_type == HamiltonianType.Rydberg:
        assert ham.factors[0].shape == (1, 2, 2, 3)
        assert ham.factors[1].shape == (3, 2, 2, 4)
        assert ham.factors[2].shape == (4, 2, 2, 3)
        assert ham.factors[3].shape == (3, 2, 2, 1)
    else:
        assert ham.factors[0].shape == (1, 2, 2, 4)
        assert ham.factors[1].shape == (4, 2, 2, 6)
        assert ham.factors[2].shape == (6, 2, 2, 4)
        assert ham.factors[3].shape == (4, 2, 2, 1)

    sv = torch.einsum("ijkl,lmno,opqr,rstu->ijmpsknqtu", *(ham.factors)).reshape(16, 16)
    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix, omega, delta, phi, hamiltonian_type=hamiltonian_type
    ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize(
    "hamiltonian_type",
    [
        HamiltonianType.Rydberg,
        HamiltonianType.XY,
    ],
)
def test_5_qubit(hamiltonian_type):
    num_gpus = 0
    omega, delta, phi = create_omega_delta_phi(5)

    interaction_matrix = torch.randn(5, 5, dtype=torch.float64)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    interaction_matrix.fill_diagonal_(0)

    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=hamiltonian_type,
        num_gpus_to_use=num_gpus,
    )
    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
    )

    if hamiltonian_type == HamiltonianType.Rydberg:
        assert ham.factors[0].shape == (1, 2, 2, 3)
        assert ham.factors[1].shape == (3, 2, 2, 4)
        assert ham.factors[2].shape == (4, 2, 2, 4)
        assert ham.factors[3].shape == (4, 2, 2, 3)
        assert ham.factors[4].shape == (3, 2, 2, 1)
    else:
        assert ham.factors[0].shape == (1, 2, 2, 4)
        assert ham.factors[1].shape == (4, 2, 2, 6)
        assert ham.factors[2].shape == (6, 2, 2, 6)
        assert ham.factors[3].shape == (6, 2, 2, 4)
        assert ham.factors[4].shape == (4, 2, 2, 1)

    sv = torch.einsum("abcd,defg,ghij,jklm,mnop->abehkncfilop", *(ham.factors)).reshape(
        32, 32
    )
    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix, omega, delta, phi, hamiltonian_type=hamiltonian_type
    ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize(
    "hamiltonian_type",
    [
        HamiltonianType.Rydberg,
        HamiltonianType.XY,
    ],
)
def test_9_qubit_noise(hamiltonian_type):
    num_gpus = 0
    omega = torch.tensor([12.566370614359172] * 9, dtype=dtype)
    delta = torch.tensor([10.771174812307862] * 9, dtype=dtype)
    phi = torch.tensor([torch.pi] * 9, dtype=dtype)

    interaction_matrix = torch.randn(9, 9, dtype=torch.float64)
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    interaction_matrix.fill_diagonal_(0)

    lindbladians = [
        torch.tensor([[-5, 4], [2, 5]], dtype=dtype),
        torch.tensor([[2, 3], [1.5, 5j]], dtype=dtype),
        torch.tensor([[-2.5j + 0.5, 2.3], [1, 2]], dtype=dtype),
    ]

    noise = compute_noise_from_lindbladians(lindbladians)

    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=hamiltonian_type,
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
    ).reshape(512, 512)

    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix, omega, delta, phi, noise, hamiltonian_type=hamiltonian_type
    ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


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


@pytest.mark.parametrize(
    "hamiltonian_type",
    [
        HamiltonianType.Rydberg,
        HamiltonianType.XY,
    ],
)
def test_truncation_random(hamiltonian_type):
    n = 9
    num_gpus = 0

    omega = torch.tensor([12.566370614359172] * n, dtype=dtype)
    delta = torch.zeros(n, dtype=dtype)
    phi = torch.zeros(n, dtype=dtype)

    interaction_matrix = torch.randn(n, n, dtype=torch.float64)
    interaction_matrix = 0.5 * (interaction_matrix + interaction_matrix.T)
    interaction_matrix.fill_diagonal_(0)

    interaction_matrix[interaction_matrix < 0.7] = 0.0
    ham = make_H(
        interaction_matrix=interaction_matrix,
        hamiltonian_type=hamiltonian_type,
        num_gpus_to_use=num_gpus,
    )
    update_H(
        hamiltonian=ham,
        omega=omega,
        delta=delta,
        phi=phi,
    )

    sv = torch.einsum(
        "abcd,defg,ghij,jklm,mnop,pqrs,stuv,vwxy,yzAB->abehknqtwzcfiloruxAB",
        *(ham.factors),
    ).reshape(1 << n, 1 << n)

    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix,
        omega,
        delta,
        phi,
        torch.zeros(2, 2),
        hamiltonian_type=hamiltonian_type,
    ).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


@pytest.mark.parametrize(
    "hamiltonian_type",
    [
        HamiltonianType.Rydberg,
        HamiltonianType.XY,
    ],
)
def test_truncation_nn(hamiltonian_type):
    n = 5

    omega = torch.zeros(n, dtype=dtype)
    delta = torch.zeros(n, dtype=dtype)
    phi = torch.zeros(n, dtype=dtype)

    interaction_matrix = torch.diag(torch.tensor([1.0] * (n - 1), dtype=torch.float64), 1)
    interaction_matrix = interaction_matrix + interaction_matrix.T
    ham = make_H(
        interaction_matrix=interaction_matrix,
        num_gpus_to_use=0,
        hamiltonian_type=hamiltonian_type,
    )

    sv = torch.einsum("abcd,defg,ghij,jklm,mnop->abehkncfilop", *(ham.factors)).reshape(
        1 << n, 1 << n
    )

    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(
        interaction_matrix,
        omega,
        delta,
        phi,
        torch.zeros(2, 2),
        hamiltonian_type=hamiltonian_type,
    ).to(dev)

    if hamiltonian_type == HamiltonianType.Rydberg:
        size = 3
    elif hamiltonian_type == HamiltonianType.XY:
        size = 4
    else:
        raise NotImplementedError("Extend the tests")

    assert ham.factors[0].shape == (1, 2, 2, size)
    assert ham.factors[1].shape == (size, 2, 2, size)
    assert ham.factors[2].shape == (size, 2, 2, size)
    assert ham.factors[3].shape == (size, 2, 2, size)
    assert ham.factors[4].shape == (size, 2, 2, 1)

    assert torch.allclose(
        sv,
        expected,
    )
