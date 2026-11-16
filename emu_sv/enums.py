from enum import Enum


class Exponentiator(str, Enum):
    """Which exponentiation algorithm to use in emu-sv.

    Members:

    - Exponentiator.DEFAULT: Use the default logic for determining the algorithm.
    - Exponentiator.LANCZOS: Use the Lanczos algorithm.
    - Exponentiator.CHEBYSHEV: Approximate the exponent using Chebyshev polynomials.
    """

    DEFAULT = "default"
    LANCZOS = "lanczos"
    CHEBYSHEV = "chebyshev"


class Solver(str, Enum):
    """Available solvers used by emu-sv for effective noise.
    Use these values to select the algorithm for time evolution.
    By default Lindblad is used if only effective noise is present,
    if shot noise is also present, emu-sv will use Monte Carlo.

    Members:

    - Solver.DEFAULT: Use the default logic for determining the solver.
    - Solver.MONTECARLO: Monte Carlo solver
    - Solver.LINDBLAD: Lindblad master equation solver
    """

    DEFAULT = "default"
    MONTECARLO = "montecarlo"
    LINDBLAD = "lindblad"


class HamiltonianFormat(str, Enum):
    """How emu-sv applies the Hamiltonian to state vectors.

    Members:

    - HamiltonianFormat.DEFAULT: CSR on CPU up to 16 qubits, MATRIX_FREE otherwise.
    - HamiltonianFormat.MATRIX_FREE: apply each term of H directly to the state,
      without storing H. On GPU, a fused kernel is used where available.
    - HamiltonianFormat.CSR: store H as a CSR sparse matrix. Faster for small
      states, but takes (N + 1) entries per basis state.

    CSR is not available for density matrices (Lindblad noise without Monte Carlo).
    """

    DEFAULT = "default"
    MATRIX_FREE = "matrix_free"
    CSR = "csr"
