import numpy as np
import qutip
import json
from pathlib import Path

from pulser import Sequence
from pulser_simulation import QutipEmulator


def run_with_pulser(
    seq: Sequence,
    *,
    timestep: float = 10.0,
    sim_config=None,
    with_modulation: bool = False,
    output_file: Path | None = None,
):
    """
    Returns a dictionary of the results of a Pulser simulation.

    Args:
        - seq: sequence to run in Pulser
        - timestep: duration of a time step in ns
        - sim_config: optional extra configuration for Pulser's QutipEmulator
        - with_modulation: if True, use hardware modulated pulses
        - output_file: an optional pathlib.Path where to store resulting observables values

    Note:
        To make benchmarks against Pulser easier, the resulting dictionary
        has the same keys as the Results object returned by `MPSBackend.run(...)`.
    """
    seq_length = seq.get_duration(include_fall_time=with_modulation)
    steps = int(seq_length / timestep)
    times = np.linspace(0, seq_length, steps + 2)

    qubit_count = len(seq.register.qubits)

    sim = QutipEmulator.from_sequence(
        seq,
        evaluation_times=times / 1000,
        with_modulation=with_modulation,
    )

    if sim_config:
        sim.set_config(sim_config)

    simulation_output = sim.run()

    # TODO: add correlations
    pulser_results = {
        "energy": {},
        "energy_variance": {},
        "qubit_density": {},
    }

    def get_density_operator(qubit_index: int, total_sites: int):
        factors = [qutip.qeye(2) for _ in range(total_sites)]
        factors[qubit_index] = qutip.ket("0") * qutip.bra("0")
        return qutip.tensor(factors)

    for i, t in enumerate(times):
        h = sim.get_hamiltonian(t)
        state = simulation_output.states[i]
        if i != len(times) - 1:
            # Avoid the last point where detuning is zero,
            # which doesn't compare well with EMU-MPS on plots.
            pulser_results["energy"][t] = qutip.expect(h, state)
            pulser_results["energy_variance"][t] = (
                qutip.expect(h.dag() * h, state) - qutip.expect(h, state) ** 2
            )
        pulser_results["qubit_density"][t] = [
            qutip.expect(get_density_operator(j, qubit_count), state)
            for j in range(qubit_count)
        ]

    if output_file is not None:
        # Save result dict to json.
        with output_file.open("w") as file_handle:
            json.dump(pulser_results, file_handle)

    return pulser_results
