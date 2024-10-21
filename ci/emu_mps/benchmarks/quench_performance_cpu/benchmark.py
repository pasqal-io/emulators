from pathlib import Path
import subprocess
from benchmarkutils.sequenceutils import make_quench_2d_seq
from benchmarkutils.parseutils import parse_latest_log, parse_benckmark_results_2d
from benchmarkutils.plotutils import plot_performance_2d_benchmark


script_dir = Path(__file__).parent
res_dir = script_dir / "results"
res_dir.mkdir(exist_ok=True)
# store all additional benchmark results in /log
log_dir = res_dir / "log"
log_dir.mkdir(exist_ok=True)

title = "Quench performance 2d - CPU"
print(f"Starting {title} benchmark")

# loop over different registers, seqs, GPU, timesteps...
Nxs = [2, 3, 4, 5]
Nys = [2, 3, 4, 5]
for Nx in Nxs:
    for Ny in Nys:
        # make Pulser sequence
        seq = make_quench_2d_seq(Nx, Ny)
        filename = f"Nx{Nx}Ny{Ny}"
        log_file = log_dir / filename
        print(f"\tRegister: {Nx}x{Ny} atoms")
        with open(log_file, "w") as f:
            subprocess.run(
                [
                    "python3",
                    str(script_dir / "emuct_run.py"),
                    seq.to_abstract_repr(),
                ],
                stdout=f,
            )

        print("\t\tParsing logfile...")
        parse_latest_log(log_file, output_name=filename)

parse_benckmark_results_2d(log_dir, Nxs, Nys)
plot_performance_2d_benchmark(log_dir, title)
