from enum import Enum


class Solver(str, Enum):
    """Solver algorithm for evolving the quantum state.

    The solver parameter selects the algorithm used to evolve the system
    using a Pulser sequence.

    Attributes:
        TDVP: Time-Dependent Variational Principle algorithm.
            Performs real-time evolution of the MPS using the two-site
            TDVP algorithm. This is the default solver.
        DMRG: Density Matrix Renormalization Group algorithm.
            Variationally minimizes the effective Hamiltonian using
            the two-site DMRG algorithm. Typically used for simulating
            adiabatic sequences.

    Examples:
        >>> from emu_mps import MPSConfig, Solver
        >>> config = MPSConfig(solver=Solver.TDVP)  # default
        >>> config = MPSConfig(solver=Solver.DMRG)  # for adiabatic sequences
    """

    TDVP = "tdvp"
    DMRG = "dmrg"
