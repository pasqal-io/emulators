from pathlib import Path
from benchmarkutils.sequenceutils import make_quench_2d_seq
import emu_mps
import multiprocessing as mp

script_dir = Path(__file__).parent
res_dir = script_dir / "results"
res_dir.mkdir(exist_ok=True)

title = "TDVP performance - GPU"
print(f"Starting {title} benchmark")

backend = emu_mps.MPSBackend()

statistics = mp.Manager().dict()


def run_simulation(n, dim):
    seq = make_quench_2d_seq(n, n)
    config = emu_mps.MPSConfig(
        num_gpus_to_use=1,
        max_bond_dim=dim,
        log_file=res_dir / f"log_{n}_{dim}.log",
        extra_krylov_tolerance=1e4,
        precision=1e-16,
    )
    results = backend.run(seq, mps_config=config)
    results.dump(res_dir / f"results_{n}_{dim}.json")
    statistics[(n, dim)] = results.statistics


N = [9, 8, 7, 6]
bond_dims = [900, 800, 700, 600, 500]
for n in N:
    for dim in bond_dims:
        # Run simulation in a separate process for correct measurement of ru_maxrss.
        process = mp.Process(target=run_simulation, args=(n, dim))
        process.start()
        process.join()
