from pathlib import Path
from benchmarkutils.sequenceutils import make_quench_2d_seq
from benchmarkutils.plotutils import plot_observables_and_performance
import emu_mps
import multiprocessing as mp

script_dir = Path(__file__).parent
res_dir = script_dir / "results"
res_dir.mkdir(exist_ok=True)

Nx = 4
Ny = 4

title = f"Quench the {Nx}x{Ny} register"
print(f"Starting {title} benchmark")

benchmark_suite = {
    "CPU": {"num_gpus_to_use": 0},
    "1-GPU": {"num_gpus_to_use": 1},
    "2-GPU": {"num_gpus_to_use": 2},
}

observables_dt = 10
backend = emu_mps.MPSBackend()

all_results = mp.Manager().dict()


def run_simulation(configuration, params):
    num_gpus_to_use = params["num_gpus_to_use"]
    seq = make_quench_2d_seq(Nx, Ny)
    evaluation_times = set(range(0, seq.get_duration(), observables_dt))

    obs = [
        emu_mps.QubitDensity(
            basis=("r", "g"),
            nqubits=len(seq.register.qubit_ids),
            evaluation_times=evaluation_times,
        ),
        emu_mps.Energy(evaluation_times=evaluation_times),
        emu_mps.EnergyVariance(evaluation_times=evaluation_times),
    ]

    config = emu_mps.MPSConfig(
        num_gpus_to_use=num_gpus_to_use,
        observables=obs,
        log_file=res_dir / f"log_{configuration}.log",
    )

    results = backend.run(seq, mps_config=config)
    results.dump(res_dir / f"results_{configuration}.json")
    all_results[configuration] = results


for configuration, params in benchmark_suite.items():
    # Run simulation in a separate process for correct measurement of ru_maxrss.
    p = mp.Process(target=run_simulation, args=(configuration, params))
    p.start()
    p.join()


print("Benchmark executed!")
print("Plotting...")
plot_observables_and_performance(all_results=all_results, title=title, output_dir=res_dir)
