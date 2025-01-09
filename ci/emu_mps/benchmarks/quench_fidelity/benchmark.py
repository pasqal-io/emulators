from pathlib import Path
from benchmarkutils.plotutils import plot_fidelity_benchmark
from benchmarkutils.pulserutils import run_with_pulser
from benchmarkutils.sequenceutils import make_quench_2d_seq

import emu_mps

script_dir = Path(__file__).parent
res_dir = script_dir / "results"
res_dir.mkdir(exist_ok=True)

title = "EMU-MPS: Quench fidelity 2d - CPU"
print(f"Starting {title} benchmark")

seq = make_quench_2d_seq(3, 3)

output_name = "quench_fidelity.json"

backend = emu_mps.MPSBackend()

emu_mps_results = {}
dts = [1, 5, 10]
precisions = [1e-8, 1e-5]
for dt in dts:  # ns
    evaluation_times = set(range(0, seq.get_duration(), dt))
    obs = [
        emu_mps.QubitDensity(
            basis=("r", "g"),
            nqubits=len(seq.register.qubit_ids),
            evaluation_times=evaluation_times,
        ),
        emu_mps.Energy(evaluation_times=evaluation_times),
        emu_mps.EnergyVariance(evaluation_times=evaluation_times),
    ]
    for j, precision in enumerate(precisions):
        config = emu_mps.MPSConfig(
            num_gpus_to_use=0,
            dt=dt,
            precision=precision,
            observables=obs,
            log_file=res_dir / f"log_dt={dt},precision={precision}.log",
        )

        results = backend.run(seq, mps_config=config)
        results.dump(res_dir / f"results_dt={dt},precision={precision}.json")
        emu_mps_results[f"dt={dt}, prec={precision}"] = results

pulser_results = {}
for dt in dts:
    results = run_with_pulser(
        seq,
        timestep=dt,
        output_file=res_dir / f"pulser_obs_{output_name}",
    )
    pulser_results[f"dt={dt}"] = results


plot_fidelity_benchmark(
    emu_mps_results=emu_mps_results,
    pulser_results=pulser_results,
    title=title,
    output_dir=res_dir,
)
