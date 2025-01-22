import torch
from emu_base.math import krylov_exp
from emu_mps import MPS, MPO, MPSConfig
from emu_mps.tdvp import (
    apply_effective_Hamiltonian,
    right_baths,
    evolve_single,
    evolve_pair,
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
    left_bath = torch.randn(4, 3, 4, dtype=torch.complex128)
    right_bath = torch.randn(7, 6, 7, dtype=torch.complex128)
    state = torch.randn(4, 4, 7, dtype=torch.complex128)
    left_ham = torch.randn(3, 2, 2, 8, dtype=torch.complex128)
    right_ham = torch.randn(8, 2, 2, 6, dtype=torch.complex128)
    ham = torch.einsum("ijkl,lmno->ijmkno", left_ham, right_ham).reshape(3, 4, 4, 6)
    actual = apply_effective_Hamiltonian(state, ham, left_bath, right_bath)
    assert actual.shape == (4, 4, 7)
    # this is the expression apply_2_site_Hamiltonian implements,
    # but doing it manually is much faster
    expected = torch.einsum(
        "pnoq,krnl,lsom,ikp,jmq->irsj",
        state.reshape(4, 2, 2, 7),
        left_ham,
        right_ham,
        left_bath,
        right_bath,
    ).reshape(4, 4, 7)
    assert torch.allclose(actual, expected)


def test_apply_1_site_effective_Hamiltonian():
    left_bath = torch.randn(4, 3, 4, dtype=torch.complex128)
    right_bath = torch.randn(7, 6, 7, dtype=torch.complex128)
    state = torch.randn(4, 2, 7, dtype=torch.complex128)
    ham = torch.randn(3, 2, 2, 6, dtype=torch.complex128)
    actual = apply_effective_Hamiltonian(state, ham, left_bath, right_bath)
    assert actual.shape == (4, 2, 7)
    # this is the expression apply_2_site_Hamiltonian implements,
    # but doing it manually is much faster
    expected = torch.einsum(
        "pnq,krnm,ikp,jmq->irj",
        state,
        ham,
        left_bath,
        right_bath,
    ).reshape(4, 2, 7)
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
    op = lambda x: torch.pi * 0.5j * apply_effective_Hamiltonian(x, ham, bath, bath)

    result = krylov_exp(op, state, exp_tolerance=1e-7, norm_tolerance=1e-12)

    # result should be |10>, and the algorithm will terminate on
    # b < norm_tolerance, since in the second iteration, b=0
    assert torch.allclose(
        result,
        torch.tensor([[[0.0], [0.0], [1.0j], [0.0]]], dtype=torch.complex128),
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
    op = lambda x: 1.0j * apply_effective_Hamiltonian(x, ham, left_bath, right_bath)

    result = krylov_exp(op, state, norm_tolerance=1e-7, exp_tolerance=1e-10)

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


def test_evolve_single():
    left_bath = torch.rand(3, 4, 3, dtype=torch.complex128)
    right_bath = torch.rand(4, 5, 4, dtype=torch.complex128)
    state_factor = torch.rand(3, 2, 4, dtype=torch.complex128)
    ham_factor = torch.rand(4, 2, 2, 5, dtype=torch.complex128)

    op = torch.einsum("abc,bdef,gfh->adgceh", left_bath, ham_factor, right_bath).reshape(
        3 * 2 * 4, -1
    )

    dt = 10

    exp_op = torch.linalg.matrix_exp(-1j * 0.001 * dt * op)

    expected = torch.tensordot(exp_op, state_factor.reshape(-1), dims=1).reshape(3, 2, 4)

    actual = evolve_single(
        state_factor=state_factor,
        baths=(left_bath, right_bath),
        ham_factor=ham_factor,
        dt=dt,
        config=MPSConfig(
            max_bond_dim=10,
        ),
        is_hermitian=False,
    )

    assert torch.allclose(expected, actual, rtol=0, atol=1e-8)


def test_evolve_pair():
    left_bath = torch.rand(3, 4, 3, dtype=torch.complex128)
    right_bath = torch.rand(5, 6, 5, dtype=torch.complex128)
    left_state_factor = torch.rand(3, 2, 4, dtype=torch.complex128)
    right_state_factor = torch.rand(4, 2, 5, dtype=torch.complex128)
    left_ham_factor = torch.rand(4, 2, 2, 5, dtype=torch.complex128)
    right_ham_factor = torch.rand(5, 2, 2, 6, dtype=torch.complex128)

    op = torch.einsum(
        "abc,bdef,fghi,jik->adgjcehk",
        left_bath,
        left_ham_factor,
        right_ham_factor,
        right_bath,
    ).reshape(3 * 2 * 2 * 5, -1)

    dt = 10

    exp_op = torch.linalg.matrix_exp(-1j * 0.001 * dt * op)

    expected = torch.tensordot(
        exp_op,
        torch.tensordot(left_state_factor, right_state_factor, dims=1).reshape(-1),
        dims=1,
    ).reshape(3, 2, 2, 5)

    actual_left, actual_right = evolve_pair(
        state_factors=[left_state_factor, right_state_factor],
        baths=(left_bath, right_bath),
        ham_factors=[left_ham_factor, right_ham_factor],
        dt=dt,
        config=MPSConfig(
            max_bond_dim=10,
        ),
        is_hermitian=False,
        orth_center_right=False,
    )

    actual = torch.tensordot(actual_left, actual_right, dims=1)

    assert torch.allclose(expected, actual, rtol=0, atol=1e-8)
