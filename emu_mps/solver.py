from enum import Enum


class Solver(str, Enum):
    """Available MPS solvers used by emu-mps. Use these values to
    select the algorithm for time evolution / simulation.
    By defatult TDVP is used. In order to use DMRG, set the
    `solver` argument of `MPSConfig` to "dmrg" or `Solver.DMRG`.

    Args:

    - Solver.TEBD: Time-Evolving Block Decimation solver.
    - Solver.TDVP: Time-Dependent Variational Principle solver.
    """

    TDVP = "tdvp"
    DMRG = "dmrg"
