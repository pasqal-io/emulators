import numpy as np
import qutip
import json

from pulser import Sequence
from pulser_simulation import QutipEmulator


def run_with_pulser(
    seq: Sequence,
    output_dir: str = "./results",
    output_name: str = "sampling.json",
    timestep: float = 10.0,
    skip_write_output: bool = True,
    hardware_modulation: bool = False,
    sim_config=None,
):
    """
    Returns the final state as a quitip QObj and a dictionary
    of the results of a pulser simulation.

    Args:
        - seq: sequence to run in `Pulser`
        - dt: timestep
        - hardware_modulation: whether or not using hardware modulated pulses

    Note:
        To make benchmarks against `Pulser` easier, the resulting dictionary
        has the same keys of the one returned by `PyEmuTN.run()`.
    """
    # build times
    # TOFIX: EmuTN and Pulser discretize pulses differently
    # so at this stage the evaluation times are arbitrary
    seq_length = seq.get_duration(include_fall_time=hardware_modulation)
    steps = int(seq_length / timestep)
    times = np.linspace(0, seq_length, steps + 2)

    # number of atoms
    N = len(seq.register.qubits)

    # pulser run
    sim = QutipEmulator.from_sequence(
        seq,
        evaluation_times=times / 1000,
        with_modulation=hardware_modulation,
    )

    if sim_config:
        sim.set_config(sim_config)

    results = sim.run()

    # final state
    psi_final = results.states[-1]

    # make dictionary of results
    E = np.zeros(len(times))
    dE = np.zeros(len(times))
    qubit_density = np.zeros((len(times), N))

    def density(j: int, total_sites: int):
        prod = [qutip.qeye(2) for _ in range(total_sites)]
        prod[j] = qutip.ket("0") * qutip.bra("0")
        return qutip.tensor(prod)

    for i, t in enumerate(times):
        h = sim.get_hamiltonian(t)
        h2 = h.dag() * h
        state = results.states[i]
        E[i] = qutip.expect(h, state)
        dE[i] = qutip.expect(h2, state) - qutip.expect(h, state) ** 2
        for j in np.arange(N):
            qubit_density[i, j] = qutip.expect(density(j, N), state)

    # TODO: add correlations
    results_dict = {
        "time": times.tolist(),
        "energy": E.tolist(),
        "varianceH": dE.tolist(),
        "qubitDensity": qubit_density.tolist(),
    }

    # save result dict to json
    if skip_write_output is False:
        path = output_dir + "/" + "pulser_obs_" + output_name
        with open(path, "w") as outfile:
            json.dump(results_dict, outfile)

    return psi_final, results_dict
