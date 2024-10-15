from pathlib import Path
from benchmarkutils.sequenceutils import make_quench_2d_seq
from benchmarkutils.plotutils import plot_performance_2d_benchmark
import emu_mps
import multiprocessing as mp

script_dir = Path(__file__).parent
res_dir = script_dir / "results"
res_dir.mkdir(exist_ok=True)

title = "Quench performance 2d - GPU"
print(f"Starting {title} benchmark")

backend = emu_mps.MPSBackend()

statistics = mp.Manager().dict()


def run_simulation(Nx, Ny):
    seq = make_quench_2d_seq(Nx, Ny)
    config = emu_mps.MPSConfig(
        num_gpus_to_use=1, max_bond_dim=2000, log_file=res_dir / f"log_{Nx}x{Ny}.log"
    )
    results = backend.run(seq, mps_config=config)
    results.dump(res_dir / f"results_{Nx}x{Ny}.json")
    statistics[(Nx, Ny)] = results.statistics


Nxs = [2, 3, 4, 5]
Nys = [2, 3, 4, 5]
for Nx in Nxs:
    for Ny in Nys:
        # Run simulation in a separate process for correct measurement of ru_maxrss.
        process = mp.Process(target=run_simulation, args=(Nx, Ny))
        process.start()
        process.join()

plot_performance_2d_benchmark(
    statistics=statistics, title=title, output=res_dir / f"{script_dir.name}.png"
)
