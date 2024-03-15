from emu_ct.tdvp import (
    left_baths,
    right_baths,
    apply_effective_Hamiltonian,
    krylov_exp,
    tdvp,
    Config,
)
from emu_ct import MPS, MPO, inner, make_H, Register
import torch


def test_left_baths_bell():
    # state = (|0> + |1>)^3 / norm
    mps_factor1 = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.complex128)
    mps_factor2 = torch.tensor(
        [[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=torch.complex128
    )
    mps_factor3 = torch.tensor([[[1], [0]], [[0], [1]]], dtype=torch.complex128)

    # Hamiltonian X1*X2*X3
    mpo_factor = torch.tensor([[[[0], [1]], [[1], [0]]]], dtype=torch.complex128)

    state = MPS([mps_factor1, mps_factor2, mps_factor3])
    obs = MPO([mpo_factor] * 3)
    for i, b in enumerate(left_baths(state, obs, 1)):
        # Because the Hamiltonian flips all the spins, the baths have shape
        # (2,1,2) and they're all pauli_x
        if i == 0:
            assert torch.allclose(
                b, torch.ones(1, 1, 1, dtype=torch.complex128, device=b.device)
            )
        else:
            assert torch.allclose(
                b,
                torch.tensor(
                    [[[0, 1]], [[1, 0]]], dtype=torch.complex128, device=b.device
                ),
            )


def test_left_baths_total_magnetization():
    # Hamiltonian Z1+Z2+Z3
    mpo_factor1 = torch.tensor(
        [[[[1, 1], [0, 0]], [[0, 0], [-1, 1]]]], dtype=torch.complex128
    )
    mpo_factor2 = torch.tensor(
        [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[1, 1], [0, 0]], [[0, 0], [-1, 1]]]],
        dtype=torch.complex128,
    )
    mpo_factor3 = torch.tensor(
        [[[[1], [0]], [[0], [1]]], [[[1], [0]], [[0], [-1]]]], dtype=torch.complex128
    )

    # state |111>
    mps_factor = torch.tensor([[[0], [1]]], dtype=torch.complex128)

    state = MPS([mps_factor] * 3)
    obs = MPO([mpo_factor1, mpo_factor2, mpo_factor3])
    baths = left_baths(state, obs, 1)
    # The baths carry the information of the magnetization, so the baths have shape
    # (1,2,1), and L_i = [-i,1], which basically counts how magnetized the bath is.
    assert torch.allclose(
        baths[0], torch.ones(1, 1, 1, dtype=torch.complex128, device=baths[0].device)
    )
    assert torch.allclose(
        baths[1],
        torch.tensor([[[-1], [1]]], dtype=torch.complex128, device=baths[1].device),
    )
    assert torch.allclose(
        baths[2],
        torch.tensor([[[-2], [1]]], dtype=torch.complex128, device=baths[2].device),
    )


def test_right_baths_bell():
    # state = (|0> + |1>)^3 / norm
    mps_factor1 = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.complex128)
    mps_factor2 = torch.tensor(
        [[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=torch.complex128
    )
    mps_factor3 = torch.tensor([[[1], [0]], [[0], [1]]], dtype=torch.complex128)

    # Hamiltonian X1*X2*X3
    mpo_factor = torch.tensor([[[[0], [1]], [[1], [0]]]], dtype=torch.complex128)

    state = MPS([mps_factor1, mps_factor2, mps_factor3])
    obs = MPO([mpo_factor] * 3)
    for i, b in enumerate(right_baths(state, obs, 1)):
        # Because the Hamiltonian flips all the spins, the baths have shape
        # (2,1,2) and they're all pauli_x
        if i == 0:
            assert torch.allclose(
                b, torch.ones(1, 1, 1, dtype=torch.complex128, device=b.device)
            )
        else:
            assert torch.allclose(
                b,
                torch.tensor(
                    [[[0, 1]], [[1, 0]]], dtype=torch.complex128, device=b.device
                ),
            )


def test_right_baths_total_magnetization():
    # Hamiltonian Z1+Z2+Z3
    mpo_factor1 = torch.tensor(
        [[[[1, 1], [0, 0]], [[0, 0], [-1, 1]]]], dtype=torch.complex128
    )
    mpo_factor2 = torch.tensor(
        [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[1, 1], [0, 0]], [[0, 0], [-1, 1]]]],
        dtype=torch.complex128,
    )
    mpo_factor3 = torch.tensor(
        [[[[1], [0]], [[0], [1]]], [[[1], [0]], [[0], [-1]]]], dtype=torch.complex128
    )

    # state |111>
    mps_factor = torch.tensor([[[0], [1]]], dtype=torch.complex128)

    state = MPS([mps_factor] * 3)
    obs = MPO([mpo_factor1, mpo_factor2, mpo_factor3])
    baths = right_baths(state, obs, 1)
    # The baths carry the information of the magnetization, so the baths have shape
    # (1,2,1), and L_i = [-i,1], which basically counts how magnetized the bath is.
    assert torch.allclose(
        baths[0], torch.ones(1, 1, 1, dtype=torch.complex128, device=baths[0].device)
    )
    assert torch.allclose(
        baths[1],
        torch.tensor([[[1], [-1]]], dtype=torch.complex128, device=baths[1].device),
    )
    assert torch.allclose(
        baths[2],
        torch.tensor([[[1], [-2]]], dtype=torch.complex128, device=baths[2].device),
    )


def test_apply_2_site_effective_Hamiltonian():
    left_bath = torch.randn(2, 3, 4, dtype=torch.complex128)
    right_bath = torch.randn(5, 6, 7, dtype=torch.complex128)
    state = torch.randn(4, 4, 7, dtype=torch.complex128)
    left_ham = torch.randn(3, 2, 2, 8, dtype=torch.complex128)
    right_ham = torch.randn(8, 2, 2, 6, dtype=torch.complex128)
    ham = torch.einsum("ijkl,lmno->ijmkno", left_ham, right_ham).reshape(3, 4, 4, 6)
    actual = apply_effective_Hamiltonian(state, ham, left_bath, right_bath)
    assert actual.shape == (2, 4, 5)
    # this is the expression apply_2_site_Hamiltonian implements,
    # but doing it manually is much faster
    expected = torch.einsum(
        "pnoq,krnl,lsom,ikp,jmq->irsj",
        state.reshape(4, 2, 2, 7),
        left_ham,
        right_ham,
        left_bath,
        right_bath,
    ).reshape(2, 4, 5)
    assert torch.allclose(actual, expected)


def test_apply_1_site_effective_Hamiltonian():
    left_bath = torch.randn(2, 3, 4, dtype=torch.complex128)
    right_bath = torch.randn(5, 6, 7, dtype=torch.complex128)
    state = torch.randn(4, 2, 7, dtype=torch.complex128)
    ham = torch.randn(3, 2, 2, 6, dtype=torch.complex128)
    actual = apply_effective_Hamiltonian(state, ham, left_bath, right_bath)
    assert actual.shape == (2, 2, 5)
    # this is the expression apply_2_site_Hamiltonian implements,
    # but doing it manually is much faster
    expected = torch.einsum(
        "pnq,krnm,ikp,jmq->irj",
        state,
        ham,
        left_bath,
        right_bath,
    ).reshape(2, 2, 5)
    assert torch.allclose(actual, expected)


def test_krylov_exp_krylov_norm_tolerance():
    # trivial bath
    bath = torch.ones(1, 2, 1, dtype=torch.complex128)

    left_ham = torch.zeros(2, 2, 2, 2, dtype=torch.complex128)
    right_ham = torch.zeros(2, 2, 2, 2, dtype=torch.complex128)
    # X on first qubit
    left_ham[0, 0, 1, 0] = 1
    left_ham[0, 1, 0, 0] = 1
    # identity on second qubit
    right_ham[0, 1, 1, 0] = 1
    right_ham[0, 0, 0, 0] = 1
    ham = torch.einsum("ijkl,lmno->ijmkno", left_ham, right_ham).reshape(2, 4, 4, 2)

    # the state |00>
    state = torch.zeros(1, 4, 1, dtype=torch.complex128)
    state[0, 0, 0] = 1

    # i.e. u = X1
    result = krylov_exp(torch.pi * 0.5j, state, ham, bath, bath)

    # result should be |10>, and the algorithm will terminate on
    # b < norm_tolerance, since in the second iteration, b=0
    assert torch.allclose(
        result, torch.tensor([[[0.0], [0.0], [1.0j], [0.0]]], dtype=torch.complex128)
    )


def test_krylov_exp_krylov_exp_tolerance():
    # the bath will shift the bath index up one for each application of H_eff
    bath_size = 100
    left_bath = torch.zeros(bath_size, 1, bath_size, dtype=torch.complex128)
    for i in range(bath_size - 1):
        left_bath[i + 1, 0, i] = 1.0
        left_bath[i, 0, i + 1] = 1.0
    right_bath = torch.ones(1, 1, 1, dtype=torch.complex128)
    # all logic for this test is in the bath, but the code only works for qubits
    # so simply do the identity on the qubits
    ham = torch.zeros(1, 2, 2, 1, dtype=torch.complex128)
    # identity on both qubit
    ham[0, 1, 1, 0] = 1
    ham[0, 0, 0, 0] = 1
    ham = torch.einsum("ijkl,lmno->ijmkno", ham, ham).reshape(1, 4, 4, 1)

    # the state 00 at the 0th bath index
    state = torch.zeros(bath_size, 4, 1, dtype=torch.complex128)
    state[0, 0, 0] = 1

    # The output state should have psi[m,:,m] = i^m/m!|00>
    # Where m-1 is the loop iteration after which the krylov algorithm terminates
    # This way we can test that the krylov exp terminates with the correct tolerance
    # on the weights of psi. Since 1/14! < 1e-10, this means we expect the algorithm
    # to terminate after step 13
    Config().set_krylov_dim(13)  # test for bounds errors in preallocated T matrix
    result = krylov_exp(1.0j, state, ham, left_bath, right_bath)

    # All the following numbers should be identically 0, since the krylov algorithm never
    # got that far
    for i in range(14, 100):
        assert result[i, 0, 0] == 0.0 + 0.0j
    # also assert that we are proportional to the 00 state for each bath index
    assert torch.allclose(
        result[:, 1:, 0], torch.zeros(bath_size, 3, dtype=torch.complex128)
    )

    # now test that the output of the algorithm approximates the exact result
    expected = torch.linalg.matrix_exp(1.0j * left_bath.reshape(bath_size, bath_size))[
        :, 0
    ]
    assert torch.allclose(result[:, 0, 0], expected)


def test_tdvp():
    # X1+X2+X3
    mpo_factor1 = torch.tensor(
        [[[[0, 1], [1, 0]], [[1, 0], [0, 1]]]], dtype=torch.complex128
    )
    mpo_factor2 = torch.tensor(
        [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]],
        dtype=torch.complex128,
    )
    mpo_factor3 = torch.tensor(
        [[[[1], [0]], [[0], [1]]], [[[0], [1]], [[1], [0]]]], dtype=torch.complex128
    )

    # state |11111>
    mps_factor = torch.tensor([[[0], [1]]], dtype=torch.complex128)

    state = MPS([mps_factor] * 5)
    obs = MPO([mpo_factor1, mpo_factor2, mpo_factor2, mpo_factor2, mpo_factor3])

    # this applies tdvp in place
    tdvp(-0.5j * torch.pi, state, obs)
    assert abs(inner(state, state) - 1) < 1e-8

    # state -i|00000>
    expected_factor = torch.tensor([[[-1.0j], [0]]], dtype=torch.complex128)
    expected = MPS([expected_factor] * 5)

    for factor in state.factors:
        assert factor.shape == (1, 2, 1)
    assert abs(inner(state, expected) - 1) < 1e-8


def test_tdvp_state_vector():
    # run this on cpu, collecting the state vector from
    # multiple devices is beside the point of the test
    Config().set_num_devices_to_use(0)
    Config().set_bond_precision(1e-10)
    nqubits = 9

    registers = []
    for i in range(3):
        for j in range(3):
            registers.append(Register(7.0 * i, 7.0 * j))
    omegas = [torch.tensor([12.566370614359172], dtype=torch.complex128)] * nqubits
    deltas = [torch.tensor([10.771174812307862], dtype=torch.complex128)] * nqubits
    ham = make_H(registers, omegas, deltas)

    # |000000000>
    state = MPS(
        [torch.tensor([1.0, 0.0]).reshape(1, 2, 1).to(dtype=torch.complex128)] * nqubits
    )

    vec = torch.einsum(
        "abtc,cdue,efvg,ghwi,ijxk,klym,mnzo,opAq,qrBs->abdfhjlnprtuvwxyzABs",
        *(ham.factors)
    ).reshape(2**nqubits, 2**nqubits)
    expected = torch.linalg.matrix_exp(-0.01j * vec)[:, 0]
    for _ in range(10):
        tdvp(-0.001j, state, ham)
    vec = torch.einsum(
        "abc,cde,efg,ghi,ijk,klm,mno,opq,qrs->abdfhjlnprs", *(state.factors)
    ).reshape(2**nqubits)
    assert (
        abs(torch.dot(vec.conj(), expected) - 1) < 4.4e-7
    )  # very dependent on bond_precision
    assert abs(torch.dot(vec.conj(), vec) - 1) < 1e-8
    assert abs(torch.dot(expected.conj(), expected) - 1) < 1e-8
