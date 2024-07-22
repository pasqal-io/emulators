import sys
import emu_ct
import pulser
import json

seq = sys.argv[1]
seq = pulser.Sequence.from_abstract_repr(seq)
dt = 10
times = [dt * (i + 1) for i in range(int(seq.get_duration() / dt))]
out_file = sys.argv[2]

obs = [
    emu_ct.QubitDensity(basis={"r", "g"}, qubits=seq.register.qubit_ids, times=times),
    emu_ct.Energy(times=times),
    emu_ct.EnergyVariance(times=times),
]

config = emu_ct.MPSConfig(num_devices_to_use=0, observables=obs)
backend = emu_ct.MPSBackend()

results = backend.run(seq, mps_config=config)
with open(out_file, "w") as file:
    json.dump(results._results, file)
