import json
import sys

import pulser

import emu_mps


seq = sys.argv[1]
seq = pulser.Sequence.from_abstract_repr(seq)
dt = 10
times = [dt * (i + 1) for i in range(int(seq.get_duration() / dt))]
out_file = sys.argv[2]

obs = [
    emu_mps.QubitDensity(basis={"r", "g"}, qubits=seq.register.qubit_ids, times=times),
    emu_mps.Energy(times=times),
    emu_mps.EnergyVariance(times=times),
]

config = emu_mps.MPSConfig(num_devices_to_use=0, observables=obs)
backend = emu_mps.MPSBackend()

results = backend.run(seq, mps_config=config)
with open(out_file, "w") as file:
    json.dump(results._results, file)
