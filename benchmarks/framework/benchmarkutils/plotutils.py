from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

labels = {
    "χ": "$\\chi$",
    "|ψ|": "$|\\psi|$ [MiB]",
    "RSS": "RSS [MB]",
    "Δt": "$\\Delta t$ [s]",
}
max_labels = {
    "χ": "max($\\chi$)",
    "|ψ|": "max($|\\psi|$) [MiB]",
    "RSS": "max(RSS) [MB]",
    "runtime": "runtime [s]",
}


def plot_performance_2d_benchmark(res_dir: Path, title: str):
    """
    Plot performance metrics of `Emu-TN` runs with different number of qubits.
    """

    fig = plt.figure(figsize=(8, 6), layout="constrained")
    fig.suptitle(title)

    subfigs = fig.subfigures(1, 2)

    # multirun max subfig
    subfigs[1].suptitle("Performance vs N")
    axs = subfigs[1].subplots(4, 1, sharex=True)

    dictname = res_dir.parent.parent.name
    with open(res_dir / f"{dictname}.json", "r") as file:
        res = json.load(file)

    labels_to_plot = ["χ", "|ψ|", "RSS", "runtime"]
    for i, label in enumerate(labels_to_plot):
        axs[i].scatter(res["N"], res[label])
        axs[i].set_ylabel(max_labels[label])
    axs[-1].set_xlabel("N")

    # single run subfig
    maxNx = res["maxNx"]
    maxNy = res["maxNy"]

    subfigs[0].suptitle(f"single run - {maxNx}x{maxNy} register")
    axs = subfigs[0].subplots(4, 1, sharex=True)

    with open(res_dir / f"Nx{maxNx}Ny{maxNy}.json", "r") as file:
        res = json.load(file)

    labels_to_plot = ["χ", "|ψ|", "RSS", "Δt"]
    steps = res["step"]
    axs[0].plot(steps, res["χ"])
    axs[1].plot(steps, res["|ψ|"])
    RSS = [el for el in res["RSS"]]
    axs[2].plot(steps, RSS)
    axs[3].plot(steps[1:-1], res["Δt"][1:-1])
    for i, key in enumerate(labels_to_plot):
        axs[i].set_ylabel(labels[key])
    axs[-1].set_xlabel("step")

    plt.savefig(res_dir.parent / (dictname + ".png"))


def plot_qubit_shuffling_benchmark(results: dict, title: str, output_dir: Path):
    """
    Plot observables and performance comparison between the results
    of simulations in `EmuTN`.
    """
    fig = plt.figure(figsize=(8, 5), layout="constrained")
    fig.suptitle(title)
    subfigs = fig.subfigures(1, 2)

    subfigs[0].suptitle("Observables")
    axs = subfigs[0].subplots(4, 1, sharex=True)
    for key, res_dict in results.items():
        obs = res_dict["observables"]
        time = obs["time"]
        axs[0].plot(time, obs["energy"], label=key)
        axs[1].plot(time, obs["varianceH"])
        qubit_density = np.matrix(obs["qubitDensity"])
        axs[2].plot(time, qubit_density.mean(1))
        i = 3  # qubit to plot
        perm_map = res_dict["perm_map"]
        axs[3].plot(time, qubit_density[:, perm_map.index(3)])

    axs[0].set_ylabel("Energy")
    axs[0].legend()
    axs[1].set_ylabel("$\\Delta E$")
    axs[2].set_ylabel("$\\langle P_{r}\\rangle$")
    axs[3].set_ylabel("$\\langle P_{r}^{" + f"{i}" + "}\\rangle$")
    axs[-1].set_xlabel("time [ns]")

    subfigs[1].suptitle("Performance")
    axs = subfigs[1].subplots(4, 1, sharex=True)
    for key, res_dict in results.items():
        perf = res_dict["performance"]
        steps = perf["step"]
        axs[0].plot(steps, perf["χ"])
        axs[1].plot(steps, perf["|ψ|"])
        RSS = [el - perf["RSS"][1] for el in perf["RSS"]]
        axs[2].plot(steps, RSS)
        axs[3].plot(steps[2:], perf["Δt"][2:])
    axs[0].set_ylabel(labels["χ"])
    axs[1].set_ylabel(labels["|ψ|"])
    axs[2].set_ylabel(labels["RSS"])
    axs[3].set_ylabel(labels["Δt"])
    axs[-1].set_xlabel("steps")

    plt.savefig(output_dir / f"{output_dir.parent.name}.png")


def plot_zero_noise_benchmark(results: dict, title: str, output_dir: Path):
    """
    Plot observables and performance comparison between the results
    of simulations in `EmuTN`.
    """
    fig = plt.figure(figsize=(8, 5), layout="constrained")
    fig.suptitle(title)
    subfigs = fig.subfigures(1, 2)

    subfigs[0].suptitle("Observables")
    axs = subfigs[0].subplots(4, 1, sharex=True)
    for key, res_dict in results.items():
        obs = res_dict["observables"]
        time = obs["time"]
        axs[0].plot(time, obs["energy"], label=key)
        axs[1].plot(time, obs["varianceH"])
        qubit_density = np.matrix(obs["qubitDensity"])
        axs[2].plot(time, qubit_density.mean(1))
        i = 1  # qubit to plot
        axs[3].plot(time, qubit_density[:, i])

    axs[0].set_ylabel("Energy")
    axs[0].legend()
    axs[1].set_ylabel("$\\Delta E$")
    axs[2].set_ylabel("$\\langle P_{r}\\rangle$")
    axs[3].set_ylabel("$\\langle P_{r}^{" + f"{i}" + "}\\rangle$")
    axs[-1].set_xlabel("time [ns]")

    subfigs[1].suptitle("Performance")
    axs = subfigs[1].subplots(4, 1, sharex=True)
    for key, res_dict in results.items():
        perf = res_dict["performance"]
        steps = perf["step"]
        axs[0].plot(steps, perf["χ"])
        axs[1].plot(steps, perf["|ψ|"])
        RSS = [el - perf["RSS"][1] for el in perf["RSS"]]
        axs[2].plot(steps, RSS)
        axs[3].plot(steps[2:], perf["Δt"][2:])
    axs[0].set_ylabel(labels["χ"])
    axs[1].set_ylabel(labels["|ψ|"])
    axs[2].set_ylabel(labels["RSS"])
    axs[3].set_ylabel(labels["Δt"])
    axs[-1].set_xlabel("steps")

    plt.savefig(output_dir / f"{output_dir.parent.name}.png")


def plot_fidelity_benchmark(
    emutn_results: dict, pulser_res: dict, title: str, output_dir: Path
):
    """
    Plot and save observables vs. time for the results of the simulation
    in `EmuTN` and in `Pulser`.
    What to plot heavily depends on the benchmark.
    """
    pulser_t = pulser_res["time"]

    fig = plt.figure(figsize=(8, 5), layout="constrained")
    fig.suptitle(title)
    subfigs = fig.subfigures(1, 2)

    # subfigs[0].suptitle("Observables")
    axs = subfigs[0].subplots(2, 1, sharex=True)

    # energies
    for label, result in emutn_results.items():
        emutn_t = list(result["energy"].keys())
        axs[0].plot(emutn_t, list(result["energy"].values()), label=label)
    axs[0].plot(pulser_t[:-1], pulser_res["energy"][:-1], label="Pulser")
    axs[0].set_ylabel("Energy")
    axs[0].legend()

    # variance
    for label, result in emutn_results.items():
        emutn_t = list(result["energy_variance"].keys())
        axs[1].plot(emutn_t, list(result["energy_variance"].values()), label=label)
    axs[1].plot(pulser_t[:-1], pulser_res["varianceH"][:-1], label="Pulser")
    axs[1].set_ylabel("$\\Delta E$")
    axs[1].set_xlabel("time [ns]")

    axs = subfigs[1].subplots(4, 1, sharex=True)
    # population
    pulser_qbitdensity = np.array(pulser_res["qubitDensity"])
    qubit_to_plot = [1, 3, 6, 9]
    for i, q in enumerate(qubit_to_plot):
        for label, result in emutn_results.items():
            emutn_t = list(result["qubit_density"].keys())
            emutn_qbitdensity = np.array(list(result["qubit_density"].values()))
            axs[i].plot(emutn_t, emutn_qbitdensity[:, q - 1])
        axs[i].plot(pulser_t, pulser_qbitdensity[:, q - 1])
        axs[i].set_ylabel(f"$\\langle P_{0}^{ {q} } \\rangle$", rotation=0)
        axs[i].yaxis.set_label_coords(0.05, 0.6)
    axs[-1].set_xlabel("time [ns]")

    plt.savefig(output_dir / f"{output_dir.parent.name}.png")
