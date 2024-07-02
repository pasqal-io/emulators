from emu_ct import make_H, QubitPosition, dist2
import torch
from functools import reduce
from emu_ct.noise import compute_noise_from_lindbladians


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


def pu(i, nqubits):
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
    return single_gate(i, nqubits, n)


def n(i, j, nqubits):
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
    matrices = [torch.eye(2, 2, dtype=dtype)] * nqubits
    matrices[i] = n
    matrices[j] = n
    return reduce(torch.kron, matrices)


TEST_C6 = 0.123456


def sv_hamiltonian(
    qubit_positions: list[QubitPosition],
    omega: list[torch.Tensor],
    delta: list[torch.Tensor],
    noise: torch.Tensor = torch.zeros(2, 2),
) -> torch.Tensor:
    n_qubits = len(qubit_positions)
    device = omega[0].device
    h = torch.zeros(2**n_qubits, 2**n_qubits, dtype=dtype, device=device)
    for i in range(n_qubits):
        h += omega[i] * sigma_x(i, n_qubits).to(dtype=dtype, device=device) / 2
        h -= delta[i] * pu(i, n_qubits).to(dtype=dtype, device=device)
        h += single_gate(i, n_qubits, noise)

        for j in range(i + 1, n_qubits):
            h += (
                TEST_C6
                * n(i, j, n_qubits).to(dtype=dtype, device=device)
                / dist2(qubit_positions[i], qubit_positions[j]) ** 3
            )
    return h


#########################################


def test_2_qubit():
    omega = [torch.tensor([2.0], dtype=dtype), torch.tensor([3.0], dtype=dtype)]
    delta = [torch.tensor([5.0], dtype=dtype), torch.tensor([7.0], dtype=dtype)]
    q = [QubitPosition(0.0, 0.0), QubitPosition(10.0, 0.0)]

    ham = make_H(q, omega, delta, c6=TEST_C6, num_devices_to_use=0)
    assert ham.factors[0].shape == (1, 2, 2, 3)
    assert ham.factors[1].shape == (3, 2, 2, 1)

    sv = torch.einsum("ijkl,lmno->ijmkno", *(ham.factors)).reshape(4, 4)
    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(q, omega, delta).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


def test_noise():
    omega = torch.tensor([0.0, 0.0], dtype=dtype)
    delta = torch.tensor([0.0, 0.0], dtype=dtype)

    # Large distance to erase the interaction
    q = [QubitPosition(0.0, 0.0), QubitPosition(1000.0, 0.0)]

    rate = 0.234
    noise = -1j / 2.0 * torch.tensor([[0, 0], [0, rate]], dtype=dtype)

    ham = make_H(q, omega, delta, c6=TEST_C6, num_devices_to_use=0, noise=noise)

    sv = torch.einsum("ijkl,lmno->ijmkno", *(ham.factors)).reshape(4, 4)
    expected = sv_hamiltonian(q, omega, delta, noise=noise).to(sv.device)

    assert abs(sv[1, 1] - (-1j / 2.0 * rate)) < 1e-10

    assert torch.allclose(
        sv,
        expected,
    )


def test_4_qubit():
    omega = []
    omega.append(torch.tensor([2.0], dtype=dtype))
    omega.append(torch.tensor([3.0], dtype=dtype))
    omega.append(torch.tensor([5.0], dtype=dtype))
    omega.append(torch.tensor([7.0], dtype=dtype))
    delta = []
    delta.append(torch.tensor([2.0], dtype=dtype))
    delta.append(torch.tensor([3.0], dtype=dtype))
    delta.append(torch.tensor([5.0], dtype=dtype))
    delta.append(torch.tensor([7.0], dtype=dtype))
    q = []
    q.append(QubitPosition(0.0, 0.0))
    q.append(QubitPosition(10.0, 0.0))
    q.append(QubitPosition(0.0, 10.0))
    q.append(QubitPosition(10.0, 10.0))

    ham = make_H(q, omega, delta, c6=TEST_C6, num_devices_to_use=0)
    assert ham.factors[0].shape == (1, 2, 2, 3)
    assert ham.factors[1].shape == (3, 2, 2, 4)
    assert ham.factors[2].shape == (4, 2, 2, 3)
    assert ham.factors[3].shape == (3, 2, 2, 1)

    sv = torch.einsum("ijkl,lmno,opqr,rstu->ijmpsknqtu", *(ham.factors)).reshape(16, 16)
    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(q, omega, delta).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


def test_5_qubit():
    omega = []
    omega.append(torch.tensor([2.0], dtype=dtype))
    omega.append(torch.tensor([3.0], dtype=dtype))
    omega.append(torch.tensor([5.0], dtype=dtype))
    omega.append(torch.tensor([7.0], dtype=dtype))
    omega.append(torch.tensor([11.0], dtype=dtype))
    delta = []
    delta.append(torch.tensor([2.0], dtype=dtype))
    delta.append(torch.tensor([3.0], dtype=dtype))
    delta.append(torch.tensor([5.0], dtype=dtype))
    delta.append(torch.tensor([7.0], dtype=dtype))
    delta.append(torch.tensor([11.0], dtype=dtype))
    q = []
    q.append(QubitPosition(0.0, 0.0))
    q.append(QubitPosition(10.0, 0.0))
    q.append(QubitPosition(0.0, 10.0))
    q.append(QubitPosition(10.0, 10.0))
    q.append(QubitPosition(5.0, 5.0))

    ham = make_H(q, omega, delta, c6=TEST_C6, num_devices_to_use=0)
    assert ham.factors[0].shape == (1, 2, 2, 3)
    assert ham.factors[1].shape == (3, 2, 2, 4)
    assert ham.factors[2].shape == (4, 2, 2, 4)
    assert ham.factors[3].shape == (4, 2, 2, 3)
    assert ham.factors[4].shape == (3, 2, 2, 1)

    sv = torch.einsum("abcd,defg,ghij,jklm,mnop->abehkncfilop", *(ham.factors)).reshape(
        32, 32
    )
    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(q, omega, delta).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


def test_9_qubit_noise():
    omega = [torch.tensor([12.566370614359172], dtype=dtype)] * 9
    delta = [torch.tensor([10.771174812307862], dtype=dtype)] * 9
    q = []
    for i in range(3):
        for j in range(3):
            q.append(QubitPosition(7.0 * i, 7.0 * j))

    lindbladians = [
        torch.tensor([[-5, 4], [2, 5]], dtype=dtype),
        torch.tensor([[2, 3], [1.5, 5j]], dtype=dtype),
        torch.tensor([[-2.5j + 0.5, 2.3], [1, 2]], dtype=dtype),
    ]

    noise = compute_noise_from_lindbladians(lindbladians)

    ham = make_H(q, omega, delta, c6=TEST_C6, num_devices_to_use=0, noise=noise)

    sv = torch.einsum(
        "abcd,defg,ghij,jklm,mnop,pqrs,stuv,vwxy,yzAB->abehknqtwzcfiloruxAB",
        *(ham.factors)
    ).reshape(512, 512)
    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(q, omega, delta, noise=noise).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


def test_differentiation():
    n = 5
    omega = torch.tensor([1.0] * n, dtype=dtype, requires_grad=True)
    delta = torch.tensor([1.0] * n, dtype=dtype, requires_grad=True)
    q = [
        QubitPosition(0.0, 0.0),
        QubitPosition(10.0, 0.0),
        QubitPosition(0.0, 10.0),
        QubitPosition(10.0, 10.0),
        QubitPosition(5.0, 5.0),
    ]

    ham = make_H(q, omega, delta, num_devices_to_use=0)

    sv = torch.einsum("abcd,defg,ghij,jklm,mnop->abehkncfilop", *(ham.factors)).reshape(
        1 << n, 1 << n
    )

    # loop over each element in the state-vector form of the hamiltonian, and assert that it depends
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
                expected_omega_diff[n - (i ^ j).bit_length()] = 0.5

            assert torch.allclose(
                torch.autograd.grad(sv[i, j].real, omega, retain_graph=True)[0],
                expected_omega_diff,
            )
