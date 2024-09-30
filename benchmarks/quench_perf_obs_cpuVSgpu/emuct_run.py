import json
import sys

import pulser

import emu_mps


seq = sys.argv[1]
seq = pulser.Sequence.from_abstract_repr(seq)
dt = 10
evaluation_times = [dt * (i + 1) for i in range(int(seq.get_duration() / dt))]
out_file = sys.argv[2]
which_device = sys.argv[3]

obs = [
    emu_mps.QubitDensity(
        basis={"r", "g"},
        nqubits=len(seq.register.qubit_ids),
        evaluation_times=evaluation_times,
    ),
    emu_mps.Energy(evaluation_times=evaluation_times),
    emu_mps.EnergyVariance(evaluation_times=evaluation_times),
]

config = emu_mps.MPSConfig(num_gpus_to_use=int(which_device), observables=obs)
backend = emu_mps.MPSBackend()

results = backend.run(seq, mps_config=config)
with open(out_file, "w") as file:
    json.dump(results._results, file)
