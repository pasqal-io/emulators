from pathlib import Path
import subprocess
import json
from random import seed, sample
from benchmarkutils.sequenceutils import make_adiabatic_afm_state_2d_seq
from benchmarkutils.parseutils import parse_latest_log
from benchmarkutils.plotutils import plot_qubit_shuffling_quench_benchmark


script_dir = Path(__file__).parent
res_dir = script_dir / "results"
res_dir.mkdir(exist_ok=True)
# store all additional benchmark results in /log
log_dir = res_dir / "log"
log_dir.mkdir(exist_ok=True)

title = "Qubit shuffling 2d - CPU"
print(f"Starting {title} benchmark")

# sequence params
Nx = 4
Ny = 4

seed(2024)
benchmark_suite = {
    "unshuffled": {"perm_map": range(Nx * Ny)},
    "shuffle": {"perm_map": sample(range(Nx * Ny), k=Nx * Ny)},
}

for output_name, res_dict in benchmark_suite.items():
    seq = make_adiabatic_afm_state_2d_seq(Nx, Ny, perm_map=res_dict["perm_map"])
    log_file = log_dir / (output_name + ".log")
    out_file = log_dir / ("obs_" + output_name + ".json")
    with open(log_file, "w") as f:
        subprocess.run(
            [
                "python3",
                str(script_dir / "emuct_run.py"),
                seq.to_abstract_repr(),
                str(
                    out_file
                ),  # The path to the output file where the observables will be stored
            ],
            stdout=f,
        )
    res_dict["performance"] = parse_latest_log(log_file)
    with open(out_file, "r") as file:
        res_dict["observables"] = json.load(file)
    out_file.unlink()

plot_qubit_shuffling_quench_benchmark(
    benchmark_suite, title, res_dir, qubit_shuffling=True
)
