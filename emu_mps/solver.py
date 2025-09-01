from enum import Enum


class Solver(str, Enum):
    TDVP = "tdvp"
    DMRG = "dmrg"
