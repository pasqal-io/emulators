from pathlib import Path
import emu_mps
import random
from benchmarkutils.sequenceutils import make_adiabatic_afm_state_2d_seq
from benchmarkutils.plotutils import plot_observables_and_performance

script_dir = Path(__file__).parent
res_dir = script_dir / "results"
res_dir.mkdir(exist_ok=True)

title = "Qubit shuffling 2d - CPU"
print(f"Starting {title} benchmark")

Nx = 4
Ny = 4

random.seed(2024)
perm_maps = {
    "unshuffled": range(Nx * Ny),
    "shuffled": random.sample(range(Nx * Ny), k=Nx * Ny),
}

all_results = {}

for test_name, perm_map in perm_maps.items():
    seq = make_adiabatic_afm_state_2d_seq(Nx, Ny, perm_map=perm_map)

    dt = 10
    evaluation_times = set(range(0, seq.get_duration(), dt))
    obs = [
        emu_mps.QubitDensity(
            basis=("r", "g"),
            nqubits=Nx * Ny,
            evaluation_times=evaluation_times,
        ),
        emu_mps.Energy(evaluation_times=evaluation_times),
        emu_mps.EnergyVariance(evaluation_times=evaluation_times),
    ]

    config = emu_mps.MPSConfig(
        num_gpus_to_use=0, observables=obs, log_file=res_dir / f"log_{test_name}.log"
    )
    backend = emu_mps.MPSBackend()

    results = backend.run(seq, mps_config=config)
    results.dump(res_dir / f"results_{test_name}.json")
    all_results[test_name] = results

plot_observables_and_performance(
    all_results=all_results, title=title, output_dir=res_dir, perm_maps=perm_maps
)
