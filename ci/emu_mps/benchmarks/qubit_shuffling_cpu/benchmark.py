from pathlib import Path
import emu_mps
import random
from benchmarkutils.sequenceutils import make_adiabatic_afm_state_2d_seq
from benchmarkutils.plotutils import plot_observables_and_performance
import multiprocessing as mp

script_dir = Path(__file__).parent
res_dir = script_dir / "results"
res_dir.mkdir(exist_ok=True)

statistics = mp.Manager().dict()

Nx = 4
Ny = 4

title = f"Qubit shuffling {Nx}x{Ny} 2d - CPU"
print(f"Starting {title} benchmark")


def run_simulation(Nx, Ny, perm_map, test_name):
    backend = emu_mps.MPSBackend()

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
    results = backend.run(seq, mps_config=config)

    results.dump(res_dir / f"results_{test_name}.json")
    statistics[test_name] = results


random.seed(2024)
perm_maps = {
    "unshuffled": range(Nx * Ny),
    "shuffled": random.sample(range(Nx * Ny), k=Nx * Ny),
}

for test_name, perm_map in perm_maps.items():
    # Run simulation in a separate process for correct measurement of ru_maxrss.
    process = mp.Process(target=run_simulation, args=(Nx, Ny, perm_map, test_name))
    process.start()
    process.join()

plot_observables_and_performance(
    all_results=statistics, title=title, output_dir=res_dir, perm_maps=perm_maps
)
