from pathlib import Path
import matplotlib.pyplot as plt
from emu_base.base_classes.results import Results
import statistics

labels = {
    "max_bond_dimension": "$\\chi$",
    "memory_footprint": "$|\\psi|$ [MiB]",
    "RSS": "RSS [MB]",
    "duration": "$\\Delta t$ [s]",
    "time": "time [ns]",
    "energy": "Energy",
    "energy_variance": "$\\Delta E$",
    "qubit_density_mean": "$\\langle P_{r}\\rangle$",
    "qubit_density": lambda qubit_index: "$\\langle P_{r}^{"
    + f"{qubit_index}"
    + "}\\rangle$",
    "step": "step",
}
max_labels = {
    "max_bond_dimension": "max($\\chi$)",
    "memory_footprint": "max($|\\psi|$) [MiB]",
    "RSS": "max(RSS) [MB]",
    "duration": "runtime [s]",
}

aggregators = {  # How to aggregate all steps of a given run.
    "max_bond_dimension": max,
    "memory_footprint": max,
    "RSS": max,
    "duration": sum,
}


def plot_performance_2d_benchmark(
    *, statistics: dict[tuple[int, int], Results], title: str, output: Path
):
    """
    Plot performance metrics of EMU-MPS runs with different number of qubits.
    """
    fig = plt.figure(figsize=(8, 6), layout="constrained")
    fig.suptitle(title)

    subfigs = fig.subfigures(1, 2)

    # multirun max subfig
    subfigs[1].suptitle("Performance vs N")
    axs = subfigs[1].subplots(4, 1, sharex=True)

    labels_to_plot = ["max_bond_dimension", "memory_footprint", "RSS", "duration"]

    ns = [nx * ny for nx, ny in statistics.keys()]
    for i, label in enumerate(labels_to_plot):
        axs[i].scatter(
            ns,
            [
                aggregators[label](
                    step_statistics[label] for step_statistics in statistic["steps"]
                )
                for statistic in statistics.values()
            ],
        )
        axs[i].set_ylabel(max_labels[label])
    axs[-1].set_xlabel("N")

    # single run subfig
    largest_grid = max(statistics.keys(), key=lambda grid_dim: grid_dim[0] * grid_dim[1])
    largest_grid_step_statistics = statistics[largest_grid]["steps"]

    subfigs[0].suptitle(f"single run - {largest_grid[0]}x{largest_grid[1]} register")
    axs = subfigs[0].subplots(4, 1, sharex=True)

    step_count = len(largest_grid_step_statistics)
    for i, label in enumerate(labels_to_plot):
        axs[i].plot(
            range(step_count),
            [step_stat[label] for step_stat in largest_grid_step_statistics],
        )
        axs[i].set_ylabel(labels[label])

    axs[-1].set_xlabel("step")

    plt.savefig(output)


def plot_fidelity_benchmark(
    *, emu_mps_results: dict, pulser_results: dict, title: str, output_dir: Path
):
    """
    Plot observables vs. time for the results of the simulation
    in EMU-MPS and in Pulser.
    """
    fig = plt.figure(figsize=(8, 5), layout="constrained")
    fig.suptitle(title)
    subfigs = fig.subfigures(1, 2)

    axs = subfigs[0].subplots(2, 1, sharex=True)

    # energy
    for label, result in emu_mps_results.items():
        times = list(result["energy"].keys())
        axs[0].plot(times, list(result["energy"].values()), label=label)
    axs[0].plot(
        pulser_results["energy"].keys(),
        pulser_results["energy"].values(),
        label="Pulser",
    )
    axs[0].set_ylabel(labels["energy"])
    axs[0].legend()

    # variance
    for label, result in emu_mps_results.items():
        times = list(result["energy_variance"].keys())
        axs[1].plot(times, list(result["energy_variance"].values()), label=label)
    axs[1].plot(
        pulser_results["energy_variance"].keys(),
        pulser_results["energy_variance"].values(),
        label="Pulser",
    )
    axs[1].set_ylabel(labels["energy_variance"])
    axs[1].set_xlabel(labels["time"])

    axs = subfigs[1].subplots(4, 1, sharex=True)

    # qubit density
    qubits_to_plot = [1, 3, 6, 9]
    for i, qubit_index in enumerate(qubits_to_plot):
        for label, result in emu_mps_results.items():
            axs[i].plot(
                result["qubit_density"].keys(),
                [
                    qubit_density[qubit_index - 1]
                    for qubit_density in result["qubit_density"].values()
                ],
                label=label,
            )
        axs[i].plot(
            pulser_results["qubit_density"].keys(),
            [
                qubit_density[qubit_index - 1]
                for qubit_density in pulser_results["qubit_density"].values()
            ],
            label="Pulser",
        )
        axs[i].set_ylabel(labels["qubit_density"](qubit_index), rotation=0)
        axs[i].yaxis.set_label_coords(0.05, 0.6)
    axs[-1].set_xlabel(labels["time"])

    plt.savefig(output_dir / f"{output_dir.parent.name}.png")


def plot_observables_and_performance(
    all_results: dict, title: str, output_dir: Path, perm_maps=None
):
    """
    Plot observables and performance comparison between the results
    of different EMU-MPS simulations.
    """
    fig = plt.figure(figsize=(10, 6), layout="constrained")
    fig.suptitle(title)
    subfigs = fig.subfigures(1, 2)

    qubit_index_for_density = 3  # single qubit to plot
    subfigs[0].suptitle("Observables")
    axs = subfigs[0].subplots(4, 1, sharex=True)

    for label, results in all_results.items():
        # Assuming all observables have same evaluation_times
        axs[0].plot(results["energy"].keys(), results["energy"].values(), label=label)
        axs[1].plot(
            results["energy_variance"].keys(),
            results["energy_variance"].values(),
            label=label,
        )
        axs[2].plot(
            results["qubit_density"].keys(),
            [
                statistics.mean(qubit_density)
                for qubit_density in results["qubit_density"].values()
            ],
        )
        axs[3].plot(
            results["qubit_density"].keys(),
            [
                qubit_density[
                    (
                        qubit_index_for_density
                        if perm_maps is None
                        else perm_maps[label].index(qubit_index_for_density)
                    )
                ]
                for qubit_density in results["qubit_density"].values()
            ],
            label=label,
        )

    # FIXME: xticks

    axs[0].set_ylabel(labels["energy"])
    axs[0].legend()
    axs[1].set_ylabel(labels["energy_variance"])
    axs[2].set_ylabel(labels["qubit_density_mean"])
    axs[3].set_ylabel(labels["qubit_density"](qubit_index_for_density))
    axs[-1].set_xlabel(labels["time"])

    subfigs[0].align_ylabels(axs)

    subfigs[1].suptitle("Performance")
    axs = subfigs[1].subplots(4, 1, sharex=True)
    for label, results in all_results.items():
        steps = results.statistics["steps"]
        xs = range(len(steps))
        axs[0].plot(xs, [step["max_bond_dimension"] for step in steps])
        axs[1].plot(xs, [step["memory_footprint"] for step in steps])
        RSS = [step["RSS"] - steps[0]["RSS"] for step in steps]
        axs[2].plot(xs, RSS)
        axs[3].plot(xs, [step["duration"] for step in steps])
    axs[0].set_ylabel(labels["max_bond_dimension"])
    axs[1].set_ylabel(labels["memory_footprint"])
    axs[2].set_ylabel(labels["RSS"])
    axs[3].set_ylabel(labels["duration"])
    axs[-1].set_xlabel(labels["step"])

    subfigs[1].align_ylabels(axs)

    plt.savefig(output_dir / f"{output_dir.parent.name}.png")
