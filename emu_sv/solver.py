from enum import Enum


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
