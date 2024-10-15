from pathlib import Path
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

title = "Adiabatic AFM state 2d Fidelity - CPU - with depolarizing noise"
print(f"Starting {title} benchmark")

rows, columns = 2, 2
qubit_count = rows * columns

seq = make_adiabatic_afm_state_2d_seq(rows, columns)
observables_dt = 10

backend = emu_mps.MPSBackend()
evaluation_times = set(range(observables_dt, seq.get_duration(), observables_dt))
observables = [
    emu_mps.Energy(evaluation_times=evaluation_times),
    emu_mps.SecondMomentOfEnergy(evaluation_times=evaluation_times),
    emu_mps.QubitDensity(evaluation_times=evaluation_times, basis=("r", "g"), nqubits=4),
]

emu_mps_results = {}
pulser_total = {}
nruns = 100  # Number of runs for aggregating monte carlo results.
dt = 5
precision = 1e-6

depolarizing_rates = [0.2, 0.5]
for depolarizing_rate in depolarizing_rates:
    noise_model = pulser.noise_model.NoiseModel(
        depolarizing_rate=depolarizing_rate,
    )

    aggregated_results = {}

    def do_run(run_index):
        config = emu_mps.MPSConfig(
            num_gpus_to_use=0,
            dt=dt,
            precision=precision,
            observables=observables,
            noise_model=noise_model,
            log_file=res_dir
            / f"log_depolarizing_rate={depolarizing_rate}_run={run_index}.log",
        )
        return backend.run(seq, config)

    processes_count = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", cpu_count())) - 1

    with Pool(processes=processes_count) as pool:
        monte_carlo_results = pool.map(do_run, range(nruns))

    aggregated_results["energy"] = {
        t: statistics.fmean(res["energy"][t] for res in monte_carlo_results)
        for t in evaluation_times
    }

    aggregated_results["second_moment_of_energy"] = {
        t: statistics.fmean(
            res["second_moment_of_energy"][t] for res in monte_carlo_results
        )
        for t in evaluation_times
    }

    aggregated_results["qubit_density"] = {
        t: tuple(
            statistics.fmean(
                res["qubit_density"][t][qubit_index] for res in monte_carlo_results
            )
            for qubit_index in range(qubit_count)
        )
        for t in evaluation_times
    }

    assert (
        aggregated_results["second_moment_of_energy"].keys()
        == aggregated_results["energy"].keys()
    )
    aggregated_results["energy_variance"] = {
        t: second_moment_of_energy - energy**2
        for (t, second_moment_of_energy), (_, energy) in zip(
            aggregated_results["second_moment_of_energy"].items(),
            aggregated_results["energy"].items(),
        )
    }

    output_name = f"afm_state_fidelity_with_noise_{depolarizing_rate}.json"
    pulser_res = run_with_pulser(
        seq,
        timestep=dt,
        sim_config=pulser_simulation.SimConfig.from_noise_model(noise_model),
        output_file=res_dir / f"pulser_obs_{output_name}",
    )

    emu_mps_results[depolarizing_rate] = aggregated_results
    pulser_total[depolarizing_rate] = pulser_res

fig = plt.figure(figsize=(8, 5), layout="constrained")
fig.suptitle(title)

axs = fig.subplots(2, 2, sharex=True)

# energies
for rate, aggregated_results in emu_mps_results.items():
    times, energies = zip(*sorted(aggregated_results["energy"].items()))
    axs[0, 0].plot(
        times,
        energies,
        label=f"EMU-MPS λ={rate}",
    )
for rate, pulser_res in pulser_total.items():
    axs[0, 0].plot(
        pulser_res["energy"].keys(),
        pulser_res["energy"].values(),
        label=f"Pulser λ={rate}",
    )
axs[0, 0].set_ylabel("Energy")
axs[0, 0].legend()

# variance
for rate, aggregated_results in emu_mps_results.items():
    times, variances = zip(*sorted(aggregated_results["energy_variance"].items()))
    axs[1, 0].plot(
        times,
        variances,
        label=f"EMU-MPS λ={rate}",
    )
for rate, pulser_res in pulser_total.items():
    axs[1, 0].plot(
        pulser_res["energy_variance"].keys(),
        pulser_res["energy_variance"].values(),
        label=f"Pulser λ={rate}",
    )
axs[1, 0].set_ylabel("$\\Delta E$")
axs[1, 0].set_xlabel("time [ns]")
axs[1, 0].legend()

# qubit density
for rate, aggregated_results in emu_mps_results.items():
    times, densities = zip(*sorted(aggregated_results["qubit_density"].items()))
    axs[0, 1].plot(
        times,
        [x[0] for x in densities],
        label=f"emumps_{rate}",
    )
    axs[1, 1].plot(
        times,
        [x[2] for x in densities],
        label=f"emumps_{rate}",
    )
for rate, pulser_res in pulser_total.items():
    axs[0, 1].plot(
        pulser_res["qubit_density"].keys(),
        [x[0] for x in pulser_res["qubit_density"].values()],
        label=f"Pulser_{rate}",
    )
    axs[1, 1].plot(
        pulser_res["qubit_density"].keys(),
        [x[2] for x in pulser_res["qubit_density"].values()],
        label=f"Pulser_{rate}",
    )
axs[0, 1].set_ylabel("q[0] density")
axs[0, 1].set_xlabel("time [ns]")
axs[0, 1].legend()
axs[1, 1].set_ylabel("q[2] density")
axs[1, 1].set_xlabel("time [ns]")
axs[1, 1].legend()

plt.savefig(res_dir / f"{res_dir.parent.name}.png")
