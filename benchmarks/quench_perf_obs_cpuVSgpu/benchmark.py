from pathlib import Path
import subprocess
import json
from benchmarkutils.sequenceutils import make_quench_2d_seq
from benchmarkutils.parseutils import parse_latest_log
from benchmarkutils.plotutils import plot_qubit_shuffling_quench_benchmark


script_dir = Path(__file__).parent
res_dir = script_dir / "results"
res_dir.mkdir(exist_ok=True)
# store all additional benchmark results in /log
log_dir = res_dir / "log"
log_dir.mkdir(exist_ok=True)

# sequence params
Nx = 4
Ny = 4

title = f"Quench the {Nx}x{Ny} register"
print(f"Starting {title} benchmark")

benchmark_suite = {
    "CPU": {"perm_map": 0},
    "1-GPU": {"perm_map": 1},
    "2-GPU": {"perm_map": 2},
}

for output_name, params in benchmark_suite.items():
    which_device = params["perm_map"]
    seq = make_quench_2d_seq(Nx, Ny)
    log_file = log_dir / (output_name + ".log")
    out_file = log_dir / ("obs_" + output_name + ".json")
    with open(log_file, "w") as f:
        subprocess.run(
            [
                "python3",
                str(script_dir / "emuct_run.py"),
                seq.to_abstract_repr(),
                str(out_file),
                str(which_device),
            ],
            stdout=f,
        )

    params["performance"] = parse_latest_log(log_file)
    with open(out_file, "r") as file:
        params["observables"] = json.load(file)
    out_file.unlink()

print("Benchmark excecuted\nPlotting...")
plot_qubit_shuffling_quench_benchmark(benchmark_suite, title, res_dir)
