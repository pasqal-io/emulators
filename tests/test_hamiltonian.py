from emu_ct import make_H, Register, Config, dist2
import torch
from functools import reduce


#########################################
# Code for building the hamiltonian in
# state vector form. For use in the tests
#########################################
def sigma_x(i: int, nqubits: int) -> torch.Tensor:
    σ_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    matrices = [identity for _ in range(nqubits)]
    matrices[i] = σ_x
    return reduce(torch.kron, matrices)


def pu(i, nqubits):
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    matrices = [identity for _ in range(nqubits)]
    matrices[i] = n
    return reduce(torch.kron, matrices)


def n(i, j, nqubits):
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    matrices = [identity for _ in range(nqubits)]
    matrices[i] = n
    matrices[j] = n
    return reduce(torch.kron, matrices)


def sv_hamiltonian(
    registers: list[Register], omega: list[torch.Tensor], delta: list[torch.Tensor]
) -> torch.Tensor:
    n_qubits = len(registers)
    dtype = omega[0].dtype
    device = omega[0].device
    h = torch.zeros(2**n_qubits, 2**n_qubits, dtype=dtype, device=device)
    c6 = Config().get_c6()
    for i in range(n_qubits):
        h += omega[i] * sigma_x(i, n_qubits).to(dtype=dtype, device=device) / 2
        h -= delta[i] * pu(i, n_qubits).to(dtype=dtype, device=device)
        for j in range(i + 1, n_qubits):
            h += (
                c6
                * n(i, j, n_qubits).to(dtype=dtype, device=device)
                / dist2(registers[i], registers[j]) ** 3
            )
    return h


#########################################


def test_2_qubit():
    dtype = torch.float64
    omega = [torch.tensor([2.0], dtype=dtype), torch.tensor([3.0], dtype=dtype)]
    delta = [torch.tensor([5.0], dtype=dtype), torch.tensor([7.0], dtype=dtype)]
    q = [Register(0.0, 0.0), Register(10.0, 0.0)]

    ham = make_H(q, omega, delta)
    assert ham.factors[0].shape == (1, 2, 2, 3)
    assert ham.factors[1].shape == (3, 2, 2, 1)

    sv = torch.einsum("ijkl,lmno->ijmkno", *(ham.factors)).reshape(4, 4)
    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(q, omega, delta).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


def test_4_qubit():
    dtype = torch.float64
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
    q.append(Register(0.0, 0.0))
    q.append(Register(10.0, 0.0))
    q.append(Register(0.0, 10.0))
    q.append(Register(10.0, 10.0))

    ham = make_H(q, omega, delta)
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
    dtype = torch.float64
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
    q.append(Register(0.0, 0.0))
    q.append(Register(10.0, 0.0))
    q.append(Register(0.0, 10.0))
    q.append(Register(10.0, 10.0))
    q.append(Register(5.0, 5.0))

    ham = make_H(q, omega, delta)
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


def test_9_qubit():
    dtype = torch.float64
    omega = [torch.tensor([12.566370614359172], dtype=dtype)] * 9
    delta = [torch.tensor([10.771174812307862], dtype=dtype)] * 9
    q = []
    for i in range(3):
        for j in range(3):
            q.append(Register(7.0 * i, 7.0 * j))

    ham = make_H(q, omega, delta)

    sv = torch.einsum(
        "abcd,defg,ghij,jklm,mnop,pqrs,stuv,vwxy,yzAB->abehknqtwzcfiloruxAB",
        *(ham.factors)
    ).reshape(512, 512)
    dev = sv.device  # could be cpu or gpu depending on Config
    expected = sv_hamiltonian(q, omega, delta).to(dev)

    assert torch.allclose(
        sv,
        expected,
    )


def test_differentiation():
    dtype = torch.float64
    omega = []
    omega.append(torch.tensor([1.0], dtype=dtype, requires_grad=True))
    omega.append(torch.tensor([1.0], dtype=dtype, requires_grad=True))
    omega.append(torch.tensor([1.0], dtype=dtype, requires_grad=True))
    omega.append(torch.tensor([1.0], dtype=dtype, requires_grad=True))
    omega.append(torch.tensor([1.0], dtype=dtype, requires_grad=True))
    delta = []
    delta.append(torch.tensor([1.0], dtype=dtype, requires_grad=True))
    delta.append(torch.tensor([1.0], dtype=dtype, requires_grad=True))
    delta.append(torch.tensor([1.0], dtype=dtype, requires_grad=True))
    delta.append(torch.tensor([1.0], dtype=dtype, requires_grad=True))
    delta.append(torch.tensor([1.0], dtype=dtype, requires_grad=True))
    q = []
    q.append(Register(0.0, 0.0))
    q.append(Register(10.0, 0.0))
    q.append(Register(0.0, 10.0))
    q.append(Register(10.0, 10.0))
    q.append(Register(5.0, 5.0))

    ham = make_H(q, omega, delta)

    sv = torch.einsum("abcd,defg,ghij,jklm,mnop->abehkncfilop", *(ham.factors)).reshape(
        32, 32
    )

    # loop over each element in the state-vector form of the hamiltonian, and assert that it depends
    # on the omegas and deltas in the correct way.
    for i in range(32):
        for j in range(32):
            i_str = format(i, "b").zfill(5)
            j_str = format(j, "b").zfill(5)
            diffs = [
                i for i, (left, right) in enumerate(zip(i_str, j_str)) if left != right
            ]
            for k in range(5):
                assert torch.allclose(
                    torch.autograd.grad(sv[i, j], delta[k], retain_graph=True)[0],
                    torch.tensor(
                        [-1.0 if i_str[-5 + k] == "1" and i == j else 0.0], dtype=dtype
                    ),
                )
                assert torch.allclose(
                    torch.autograd.grad(sv[i, j], omega[k], retain_graph=True)[0],
                    torch.tensor([0.5 if diffs == [k] else 0.0], dtype=dtype),
                )
