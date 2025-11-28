"""
Tutorial: Creating Custom Observables with the Rydberg Basis ("g", "r")
=======================================================================

This tutorial explains how to create custom observables for measuring
expectation values during quantum simulations using emu_mps.

There are TWO main approaches to define operators for the Expectation
observable:

1. MPO.from_operator_repr(): Use symbolic notation with basis state labels
2. Direct MPO construction: Build tensors manually with matrix elements

Both methods work with the Rydberg basis (with leakage you add the"x" and XY
you can do it with "1" and "0" ) where:

- |g> (ground state) is represented as [1, 0]^T
- |r> (Rydberg state) is represented as [0, 1]^T

The matrix element notation "ab" (e.g., "rg", "gr") corresponds to |a><b|.
"""

import torch
from math import pi

from pulser import Register
from pulser.pulse import Pulse
from pulser.sequence import Sequence
from pulser.devices import MockDevice

from emu_mps import MPO, MPSBackend, MPSConfig, Expectation

dtype = torch.complex128  # emu_mps base dtype for MPS and MPO

# =============================================================================
# SETUP: Create a simple 2-atom constant pulse
# =============================================================================

natoms = 2
reg = Register.rectangle(1, natoms, spacing=10000.0, prefix="q")

seq = Sequence(reg, MockDevice)
seq.declare_channel("ch0", "rydberg_global")
duration = 500
pulse = Pulse.ConstantPulse(duration, 2 * pi, 0.0, 0.0)
seq.add(pulse, "ch0")


# =============================================================================
# APPROACH 1: Using MPO.from_operator_repr() (Symbolic Notation)
# =============================================================================
#
# This is the recommended approach for most use cases. You define operators
# using symbolic notation based on the basis states.
#
# The `operations` parameter has the structure:
#   operations = [(coefficient, [(operator_dict, target_qubits), ...])]
#
# Where:
#   - coefficient: a scalar multiplying the entire term
#   - operator_dict: defines the single-qubit operator using basis labels
#   - target_qubits: list of qubit indices where the operator acts
#
# The operator_dict uses the notation {"ab": value} meaning: value * |a><b|
#
# For the Rydberg basis ("g", "r"):
#   - "gg" = |g><g| (projection onto ground state)
#   - "rr" = |r><r| (projection onto Rydberg state, i.e., n operator)
#   - "gr" = |g><r| (lowering operator, sigma-)
#   - "rg" = |r><g| (raising operator, sigma+)
#
# Common operators in terms of these basis elements:
#   - Pauli X: {"rg": 1.0, "gr": 1.0}
#   - Pauli Y: {"rg": -1j, "gr": 1j}
#   - Pauli Z: {"gg": 1.0, "rr": -1.0}
#   - Number operator (n): {"rr": 1.0}
#   - Identity: {"gg": 1.0, "rr": 1.0}


print("=" * 70)
print("APPROACH 1: Using MPO.from_operator_repr() (Symbolic Notation)")
print("=" * 70)

# Example 1.1: Pauli X on qubit 0
# X = |r><g| + |g><r|
operations_X0 = [(1.0, [({"rg": 1.0, "gr": 1.0}, [0])])]
X0 = MPO.from_operator_repr(
    eigenstates=("g", "r"), n_qudits=natoms, operations=operations_X0
)
print("\n1.1 Pauli X on qubit 0:")
print(f"    operations = {operations_X0}")
print(f"    Shape of first factor: {X0.factors[0].shape}")

# Example 1.2: Tensor product X0 x X1 (with coefficient 2)
# This measures correlations in the X basis
operations_XX = [(2.0, [({"rg": 1.0, "gr": 1.0}, [0, 1])])]
XX = MPO.from_operator_repr(
    eigenstates=("g", "r"), n_qudits=natoms, operations=operations_XX
)
print("\n1.2 Tensor product 2 * X0 x X1:")
print(f"    operations = {operations_XX}")

# Example 1.3: Number operator on qubit 1 (n1 = |r><r|_1 = I ⊗ |r><r| )
operations_n1 = [(1.0, [({"rr": 1.0}, [1])])]
n1 = MPO.from_operator_repr(
    eigenstates=("g", "r"), n_qudits=natoms, operations=operations_n1
)
print("\n1.3 Number operator on qubit 1 (n1 = |r><r|_1):")
print(f"    operations = {operations_n1}")

# Example 1.4: Sum of local operators: X0 + X1
# Each term in the list is a separate operator that gets summed
operations_X0_plus_X1 = [
    (1.0, [({"rg": 1.0, "gr": 1.0}, [0])]),  # X on qubit 0
    (1.0, [({"rg": 1.0, "gr": 1.0}, [1])]),  # X on qubit 1
]
X0_plus_X1 = MPO.from_operator_repr(
    eigenstates=("g", "r"), n_qudits=natoms, operations=operations_X0_plus_X1
)
print("\n1.4 Sum of operators X0 + X1:")
print(f"    operations = {operations_X0_plus_X1}")

# Example 1.5: Pauli Z on qubit 0 (Z0 = |g><g| - |r><r|)
operations_Z0 = [(1.0, [({"gg": 1.0, "rr": -1.0}, [0])])]
Z0 = MPO.from_operator_repr(
    eigenstates=("g", "r"), n_qudits=natoms, operations=operations_Z0
)
print("\n1.5 Pauli Z on qubit 0 (Z0 = |g><g| - |r><r|):")
print(f"    operations = {operations_Z0}")

# Example 1.6: Complex operator - different operators on different qubits
# X0 x Z1 (X on qubit 0, Z on qubit 1)
operations_X0_Z1 = [
    (
        1.0,
        [
            ({"rg": 1.0, "gr": 1.0}, [0]),  # X on qubit 0
            ({"gg": 1.0, "rr": -1.0}, [1]),  # Z on qubit 1
        ],
    )
]
X0_Z1 = MPO.from_operator_repr(
    eigenstates=("g", "r"), n_qudits=natoms, operations=operations_X0_Z1
)
print("\n1.6 Mixed tensor product X0 x Z1:")
print(f"    operations = {operations_X0_Z1}")


# =============================================================================
# APPROACH 2: Direct MPO Construction (Manual Tensor Definition)
# =============================================================================
#
# For more control, you can build the MPO tensors directly.
#
# Each MPO factor is a 4D tensor with shape:
#   (left_bond, output, input, right_bond)
# For single-site operators without entanglement: shape (1, d, d, 1), d=2
# (in case of leakage d=3 and in case of XY d = 2).
#
# The matrix is embedded as:
#   tensor[0, :, :, 0] = matrix
#
# In the Rydberg basis, the matrix indices are:
#   - index 0 corresponds to |g>
#   - index 1 corresponds to |r>
#
# So a matrix [[a, b], [c, d]] represents:
#   a*|g><g| + b*|g><r| + c*|r><g| + d*|r><r|

print("\n" + "=" * 70)
print("APPROACH 2: Direct MPO Construction (Manual Tensor Definition)")
print("=" * 70)

# Example 2.1: Number operator |r><r| as a tensor
# Matrix: [[0, 0], [0, 1]] because |r><r| has 1 at position (1,1)
n_matrix = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
n_tensor = n_matrix.reshape(1, 2, 2, 1)

# Identity for qubits where we don't want to measure
identity_matrix = torch.eye(2, dtype=dtype)
identity_tensor = identity_matrix.reshape(1, 2, 2, 1)

# n0 ⊗ I1 (number operator on qubit 0, identity on qubit 1)
n0_mpo = MPO([n_tensor, identity_tensor])
print("\n2.1 Number operator on qubit 0 (direct construction):")
print(f"    n_matrix =\n{n_matrix}")

# Example 2.2: Pauli X operator
# X = |r><g| + |g><r| = [[0, 1], [1, 0]]
X_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
X_tensor = X_matrix.reshape(1, 2, 2, 1)

# X0 ⊗ X1
XX_direct = MPO([X_tensor, X_tensor.clone()])
print("\n2.2 X0 x X1 (direct construction):")
print(f"    X_matrix =\n{X_matrix}")

# Example 2.3: Projection operators
# |g><g| projects onto ground state
proj_g_matrix = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=dtype)
proj_g_tensor = proj_g_matrix.reshape(1, 2, 2, 1)

# |r><r| projects onto Rydberg state
proj_r_matrix = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
proj_r_tensor = proj_r_matrix.reshape(1, 2, 2, 1)

# |gg><gg| = |g><g|_0 ⊗ |g><g|_1
proj_gg = MPO([proj_g_tensor, proj_g_tensor.clone()])

# |rr><rr| = |r><r|_0 ⊗ |r><r|_1
proj_rr = MPO([proj_r_tensor, proj_r_tensor.clone()])

print("\n2.3 Projection operators:")
print(f"    |g><g| =\n{proj_g_matrix}")
print(f"    |r><r| =\n{proj_r_matrix}")

# Example 2.4: Raising and lowering operators
# sigma+ = |r><g| (raises from g to r)
sigma_plus_matrix = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)

# sigma- = |g><r| (lowers from r to g)
sigma_minus_matrix = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)

print("\n2.4 Raising and lowering operators:")
print(f"    sigma+ = |r><g| =\n{sigma_plus_matrix}")
print(f"    sigma- = |g><r| =\n{sigma_minus_matrix}")

# Example 2.5: Combining MPOs with addition and scalar multiplication
# You can add MPOs and multiply by scalars
# Example: (|gg><gg| + |rr><rr|) measures population in |gg> and |rr>
combined_proj = proj_gg + proj_rr
print("\n2.5 Combined projector |gg><gg| + |rr><rr| created via MPO addition")

# Scalar multiplication: 2 * |rr><rr|
scaled_proj = 2.0 * proj_rr
print("    Scaled projector: 2 * |rr><rr| via scalar multiplication")


# =============================================================================
# VERIFICATION: Comparing Both Approaches
# =============================================================================
#
# Let's verify that both approaches produce the same MPO tensors.
# This is important to ensure consistency between symbolic and direct methods.

print("\n" + "=" * 70)
print("VERIFICATION: Comparing Both Approaches")
print("=" * 70)


def assert_mpo_equal(mpo1: MPO, mpo2: MPO, name: str, atol: float = 1e-10) -> None:
    """Assert that two MPOs have identical factors (up to numerical tolerance)."""
    assert len(mpo1.factors) == len(mpo2.factors), f"{name}: Different number of factors"
    for i, (f1, f2) in enumerate(zip(mpo1.factors, mpo2.factors)):
        assert (
            f1.shape == f2.shape
        ), f"{name}: Factor {i} has different shapes: {f1.shape} vs {f2.shape}"
        assert torch.allclose(
            f1, f2, atol=atol
        ), f"{name}: Factor {i} differs:\n{f1}\nvs\n{f2}"
    print(f"  [PASS] {name}")


# Build direct MPOs to compare with symbolic ones

# 1. Compare X0 (Pauli X on qubit 0)
X0_direct = MPO([X_tensor.clone(), identity_tensor.clone()])
assert_mpo_equal(X0, X0_direct, "X0 (Pauli X on qubit 0)")

# 2. Compare XX (X0 ⊗ X1 with coefficient 2)
XX_direct_scaled = 2.0 * MPO([X_tensor.clone(), X_tensor.clone()])
assert_mpo_equal(XX, XX_direct_scaled, "2 * X0 x X1")

# 3. Compare n1 (number operator on qubit 1)
n1_direct = MPO([identity_tensor.clone(), n_tensor.clone()])
assert_mpo_equal(n1, n1_direct, "n1 (number operator on qubit 1)")

# 4. Compare Z0 (Pauli Z on qubit 0)
Z_matrix = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
Z_tensor = Z_matrix.reshape(1, 2, 2, 1)
Z0_direct = MPO([Z_tensor.clone(), identity_tensor.clone()])
assert_mpo_equal(Z0, Z0_direct, "Z0 (Pauli Z on qubit 0)")

# 5. Compare X0_Z1 (X on qubit 0, Z on qubit 1)
X0_Z1_direct = MPO([X_tensor.clone(), Z_tensor.clone()])
assert_mpo_equal(X0_Z1, X0_Z1_direct, "X0 x Z1")

# 6. Compare |rr><rr| projection
proj_rr_symbolic = MPO.from_operator_repr(
    eigenstates=("g", "r"),
    n_qudits=natoms,
    operations=[(1.0, [({"rr": 1.0}, [0, 1])])],
)
assert_mpo_equal(proj_rr_symbolic, proj_rr, "|rr><rr| projection")

# 7. Compare |gg><gg| projection
proj_gg_symbolic = MPO.from_operator_repr(
    eigenstates=("g", "r"),
    n_qudits=natoms,
    operations=[(1.0, [({"gg": 1.0}, [0, 1])])],
)
assert_mpo_equal(proj_gg_symbolic, proj_gg, "|gg><gg| projection")

print("\nAll MPO comparisons passed! Both approaches produce identical results.")


# =============================================================================
# RUNNING A SIMULATION WITH CUSTOM OBSERVABLES
# =============================================================================

print("\n" + "=" * 70)
print("RUNNING SIMULATION WITH CUSTOM OBSERVABLES")
print("=" * 70)

# Create several operators to measure
# 1. X0 ⊗ X1 using symbolic notation
XX_symbolic = MPO.from_operator_repr(
    eigenstates=("g", "r"),
    n_qudits=natoms,
    operations=[(1.0, [({"rg": 1.0, "gr": 1.0}, [0, 1])])],
)

# 2. Total number operator n0 + n1 (counts total Rydberg excitations)
total_n = MPO.from_operator_repr(
    eigenstates=("g", "r"),
    n_qudits=natoms,
    operations=[
        (1.0, [({"rr": 1.0}, [0])]),
        (1.0, [({"rr": 1.0}, [1])]),
    ],
)

# 3. |rr><rr| using direct construction (prob. of both atoms in Rydberg)
proj_rr_direct = MPO([proj_r_tensor, proj_r_tensor.clone()])

# Configure simulation with observables
dt = 10
eval_times = [0.5, 1.0]  # Evaluate at 50% and 100% of the simulation

config = MPSConfig(
    num_gpus_to_use=0,
    dt=dt,
    observables=[
        Expectation(operator=XX_symbolic, evaluation_times=eval_times, tag_suffix="XX"),
        Expectation(operator=total_n, evaluation_times=eval_times, tag_suffix="total_n"),
        Expectation(
            operator=proj_rr_direct,
            evaluation_times=eval_times,
            tag_suffix="rr_population",
        ),
    ],
    optimize_qubit_ordering=False,  # Required for custom Expectation observables
)

# Run simulation
simul = MPSBackend(seq, config=config)
results = simul.run()

# Print results
print("\nResults:")
print(f"  <X0 X1> at eval times: {results.expectation_XX}")
print(f"  <n0 + n1> (total excitations): {results.expectation_total_n}")
print(f"  |<rr|psi>|^2 (both Rydberg): {results.expectation_rr_population}")


# =============================================================================
# SUMMARY: Quick Reference
# =============================================================================

print("\n" + "=" * 70)
print("QUICK REFERENCE: Operator Dictionary for Rydberg Basis")
print("=" * 70)
print(
    """
Basis: eigenstates=("g", "r")
  |g> = ground state = [1, 0]^T (index 0)
  |r> = Rydberg state = [0, 1]^T (index 1)

Symbolic notation (for MPO.from_operator_repr):
  "gg" = |g><g| = [[1,0],[0,0]]  (ground state projector)
  "rr" = |r><r| = [[0,0],[0,1]]  (Rydberg projector / number operator)
  "gr" = |g><r| = [[0,1],[0,0]]  (lowering operator sigma-)
  "rg" = |r><g| = [[0,0],[1,0]]  (raising operator sigma+)

Common operators:
  Identity (I):     {"gg": 1.0, "rr": 1.0}
  Pauli X:          {"rg": 1.0, "gr": 1.0}
  Pauli Y:          {"rg": -1j, "gr": 1j}
  Pauli Z:          {"gg": 1.0, "rr": -1.0}
  Number (n):       {"rr": 1.0}
  sigma+:           {"rg": 1.0}
  sigma-:           {"gr": 1.0}

Operations structure:
  operations = [(coeff, [(op_dict, [qubit_indices]), ...]), ...]

Example - X on qubit 0, Z on qubit 1:
  operations = [(1.0, [({"rg": 1.0, "gr": 1.0}, [0]),
                       ({"gg": 1.0, "rr": -1.0}, [1])])]

Example - Sum X0 + X1:
  operations = [(1.0, [({"rg": 1.0, "gr": 1.0}, [0])]),
                (1.0, [({"rg": 1.0, "gr": 1.0}, [1])])]
"""
)
