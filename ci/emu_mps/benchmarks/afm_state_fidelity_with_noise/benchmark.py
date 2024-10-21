from pathlib import Path
import pulser
import pulser_simulation
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from benchmarkutils.pulserutils import run_with_pulser
from benchmarkutils.sequenceutils import make_adiabatic_afm_state_2d_seq
import torch
import emu_mps
import os
import statistics

import pulser.noise_model

torch.set_num_threads(1)

script_dir = Path(__file__).parent
res_dir = script_dir / "results"
res_dir.mkdir(exist_ok=True)
# store all additional benchmark results in /log
log_dir = res_dir / "log"
log_dir.mkdir(exist_ok=True)

title = "Adiabatic AFM state 2d Fidelity - CPU - with depolarizing noise"
print(f"Starting {title} benchmark")

try:
    rows, columns = 2, 2
    qubit_count = rows * columns

    seq = make_adiabatic_afm_state_2d_seq(rows, columns)
    observables_dt = 10.0
    output_name = "afm_state_fidelity_with_noise.json"

    backend = emu_mps.MPSBackend()
    emumps_sample_times = [
        observables_dt * (i + 1) for i in range(int(seq.get_duration() / observables_dt))
    ]
    obs = [
        emu_mps.Energy(evaluation_times=emumps_sample_times),
        emu_mps.SecondMomentOfEnergy(evaluation_times=emumps_sample_times),
        emu_mps.QubitDensity(
            evaluation_times=emumps_sample_times, basis={"r", "g"}, nqubits=4
        ),
    ]

    emumps_total = {}
    pulser_total = {}
    nruns = 100  # Number of runs for aggregating monte carlo results.
    dt = 5
    precision = 1e-6

    depolarizing_rates = [0.2, 0.5]
    for depolarizing_rate in depolarizing_rates:
        noise_model = pulser.noise_model.NoiseModel(
            depolarizing_rate=depolarizing_rate,
        )

        config = emu_mps.MPSConfig(
            num_gpus_to_use=0,
            dt=dt,
            precision=precision,
            observables=obs,
            noise_model=noise_model,
        )

        emumps_res = {}

        def do_run(run_index):
            return backend.run(seq, config)

        processes_count = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", cpu_count())) - 1

        with Pool(processes=processes_count) as pool:
            results = pool.map(do_run, range(nruns))

        emumps_res["energy"] = [
            statistics.fmean(res["energy"][t] for res in results)
            for t in emumps_sample_times
        ]

        emumps_res["second_moment_of_energy"] = [
            statistics.fmean(res["second_moment_of_energy"][t] for res in results)
            for t in emumps_sample_times
        ]

        emumps_res["qubit_density"] = [
            tuple(
                statistics.fmean(res["qubit_density"][t][qubit_index] for res in results)
                for qubit_index in range(qubit_count)
            )
            for t in emumps_sample_times
        ]

        emumps_res["energy_variance"] = [
            second_moment_of_energy - energy**2
            for second_moment_of_energy, energy in zip(
                emumps_res["second_moment_of_energy"], emumps_res["energy"]
            )
        ]

        _, pulser_res = run_with_pulser(
            seq,
            output_dir=str(log_dir),
            output_name=output_name,
            timestep=dt,
            skip_write_output=False,
            sim_config=pulser_simulation.SimConfig.from_noise_model(noise_model),
        )

        emumps_total[depolarizing_rate] = emumps_res
        pulser_total[depolarizing_rate] = pulser_res

    pulser_t = pulser_total[depolarizing_rates[0]]["time"][:-1]

    fig = plt.figure(figsize=(8, 5), layout="constrained")
    fig.suptitle(title)

    axs = fig.subplots(2, 2, sharex=True)

    # energies
    for rate, emumps_res in emumps_total.items():
        axs[0, 0].plot(emumps_sample_times, emumps_res["energy"], label=f"emumps_{rate}")
    for rate, pulser_res in pulser_total.items():
        axs[0, 0].plot(pulser_t, pulser_res["energy"][:-1], label=f"Pulser_{rate}")
    axs[0, 0].set_ylabel("Energy")
    axs[0, 0].legend()

    # variance
    for rate, emumps_res in emumps_total.items():
        axs[1, 0].plot(
            emumps_sample_times, emumps_res["energy_variance"], label=f"emumps_{rate}"
        )
    for rate, pulser_res in pulser_total.items():
        axs[1, 0].plot(pulser_t, pulser_res["varianceH"][:-1], label=f"Pulser_{rate}")
    axs[1, 0].set_ylabel("$\\Delta E$")
    axs[1, 0].set_xlabel("time [ns]")
    axs[1, 0].legend()

    # qubit density
    for rate, emumps_res in emumps_total.items():
        axs[0, 1].plot(
            emumps_sample_times,
            [x[0] for x in emumps_res["qubit_density"]],
            label=f"emumps_{rate}",
        )
    for rate, pulser_res in pulser_total.items():
        axs[0, 1].plot(
            pulser_t,
            [x[0] for x in pulser_res["qubitDensity"]][:-1],
            label=f"Pulser_{rate}",
        )
    axs[0, 1].set_ylabel("q[0] density")
    axs[0, 1].set_xlabel("time [ns]")
    axs[0, 1].legend()

    for rate, emumps_res in emumps_total.items():
        axs[1, 1].plot(
            emumps_sample_times,
            [x[2] for x in emumps_res["qubit_density"]],
            label=f"emumps_{rate}",
        )
    for rate, pulser_res in pulser_total.items():
        axs[1, 1].plot(
            pulser_t,
            [x[2] for x in pulser_res["qubitDensity"]][:-1],
            label=f"Pulser_{rate}",
        )
    axs[1, 1].set_ylabel("q[2] density")
    axs[1, 1].set_xlabel("time [ns]")
    axs[1, 1].legend()

    plt.savefig(res_dir / f"{res_dir.parent.name}.png")

except Exception:
    raise
finally:
    Path.touch(res_dir / "DONE")
    print(f"{title} benchmark DONE!")
