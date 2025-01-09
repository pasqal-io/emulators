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
    "error_qubit_density": lambda qubit_index: "$\\epsilon_{\\langle P_{r}^{"
    + f"{qubit_index}"
    + "}\\rangle}$",
    "error_energy": "$\\epsilon_{E}$",
    "error_energy_variance": "$\\epsilon_{\\Delta E}$",
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

    # fmt: off
    largest_grid = max(statistics.keys(), key=lambda grid_dim: grid_dim[0] * grid_dim[1])
    # fmt: on

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


def extract_qubit_density(results: dict, qubit_index: int):
    """Extracts qubit density for a specific qubit index."""
    return [
        [density[qubit_index - 1] for density in result["qubit_density"].values()]
        for result in results.values()
    ]


def calculate_difference(emu_mps_data: list, pulser_data: list):
    """Calculates the absolute difference between emu_mps data and pulser data."""
    # NOTE: pulser data is taken at the end
    return [abs(a - b) for a, b in zip(emu_mps_data, pulser_data[1:])]


def plot_observable(
    ax,
    results: dict,
    pulser_results: dict,
    key: str,
    ylabel,
    xlabel=None,
    legend=False,
):
    """Plots an observable (e.g., energy, variance) on a given axis."""
    for label, result in results.items():
        times = list(result[key].keys())
        ax.plot(times, list(result[key].values()), label=label)

    for dt, result in pulser_results.items():
        ax.plot(
            list(result[key].keys()),
            list(result[key].values()),
            label=f"Pulser-{dt}",
        )

    ax.set_ylabel(ylabel, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if legend:
        ax.legend(loc="upper left")


def plot_observable_difference(
    ax,
    results: dict,
    pulser_results: dict,
    key: str,
    ylabel,
    xlabel=None,
    legend=False,
):
    """Plots an observable (e.g., energy, variance) on a given axis."""

    for num1, pulser_idx in enumerate(pulser_results):
        for num2, emu_mps_idx in enumerate(results):
            if (
                (num1 == 0 and num2 == 0)
                or (num1 == 0 and num2 == 1)
                or (num1 == 1 and num2 == 2)
                or (num1 == 1 and num2 == 3)
            ):
                difference = calculate_difference(
                    list(results[emu_mps_idx][key].values()),
                    list(pulser_results[pulser_idx][key].values()),
                )
                ax.plot(
                    list(results[emu_mps_idx][key].keys()),
                    difference,
                    label=list(results.keys())[num2],
                )
    ax.set_ylabel(ylabel, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if legend:
        ax.legend(loc="upper left")


def plot_qubit_density(
    axs, qubits_to_plot: list, emu_mps_results: dict, pulser_results: dict
):
    """Plots qubit densities for the selected qubits."""
    for i, qubit_index in enumerate(qubits_to_plot):
        for label, result in emu_mps_results.items():
            ax = axs[i]
            ax.plot(
                list(result["qubit_density"].keys()),
                [
                    density[qubit_index - 1]
                    for density in result["qubit_density"].values()
                ],
                label=label,
            )
        for dt, result in pulser_results.items():
            ax.plot(
                list(result["qubit_density"].keys()),
                [
                    density[qubit_index - 1]
                    for density in result["qubit_density"].values()
                ],
                label=f"Pulser-{dt}",
            )
        ax.set_ylabel(labels["qubit_density"](qubit_index), rotation=0, fontsize=14)
        ax.yaxis.set_label_coords(-0.15, 0.6)
    axs[-1].set_xlabel(labels["time"], fontsize=12)


def plot_density_differences(
    axs, qubits_to_plot: list, emu_mps_results: dict, pulser_results: dict
):
    """Plots differences between EMU-MPS and Pulser for qubit densities."""
    for i, qubit_index in enumerate(qubits_to_plot):
        emu_mps_data = extract_qubit_density(emu_mps_results, qubit_index)
        pulser_data = extract_qubit_density(pulser_results, qubit_index)
        times = [
            list(result["qubit_density"].keys()) for result in emu_mps_results.values()
        ]
        for pulser_idx in range(len(pulser_data)):
            for emu_mps_idx in range(2 * pulser_idx, 2 * pulser_idx + 2):
                difference = calculate_difference(
                    emu_mps_data[emu_mps_idx],
                    pulser_data[pulser_idx],
                )
                axs[i].plot(
                    times[emu_mps_idx],
                    difference,
                    label=list(emu_mps_results.keys())[emu_mps_idx],
                )
        axs[i].set_ylabel(
            labels["error_qubit_density"](qubit_index), rotation=0, fontsize=14
        )
        axs[i].yaxis.set_label_coords(-0.17, 0.6)
    axs[-1].set_xlabel(labels["time"], fontsize=12)


def plot_fidelity_benchmark(
    *, emu_mps_results: dict, pulser_results: dict, title: str, output_dir: Path
):
    """
    Plot observables vs. time and obsevables errors vs. time for the results of
    the simulation in EMU-Emulator and in Pulser.
    """

    # start the plot
    fig = plt.figure(figsize=(12, 14), layout="constrained")
    fig.suptitle(title, fontsize=18)
    subfigs = fig.subfigures(1, 2)

    # energy and variance plots
    axs_energy_variance = subfigs[0].subplots(5, 1, sharex=True)

    plot_observable(
        axs_energy_variance[0],
        emu_mps_results,
        pulser_results,
        "energy",
        labels["energy"],
        legend=True,
    )
    plot_observable(
        axs_energy_variance[1],
        emu_mps_results,
        pulser_results,
        "energy_variance",
        labels["energy_variance"],
    )

    # plot_difference of observable
    axs_energy_variance_diff = subfigs[1].subplots(5, 1, sharex=True)
    plot_observable_difference(
        axs_energy_variance_diff[0],
        emu_mps_results,
        pulser_results,
        "energy",
        labels["error_energy"],
    )

    plot_observable_difference(
        axs_energy_variance_diff[1],
        emu_mps_results,
        pulser_results,
        "energy_variance",
        labels["error_energy_variance"],
    )

    # qubit density plots
    qubits_to_plot = [1, 3, 6]
    plot_qubit_density(
        axs_energy_variance[2:], qubits_to_plot, emu_mps_results, pulser_results
    )

    # density difference plots
    plot_density_differences(
        axs_energy_variance_diff[2:], qubits_to_plot, emu_mps_results, pulser_results
    )

    # save the plot
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


def plot_fidelity_benchmark_emu_sv(
    *, emu_sv_results: dict, pulser_results: dict, title: str, output_dir: Path
):
    """
    Plot observables vs. time for the results of the simulation
    in EMU-SV and in Pulser.
    """

    fig = plt.figure(figsize=(12, 10), layout="constrained")
    fig.suptitle(title)
    subfigs = fig.subfigures(1, 2)

    # Qubit density plots
    axs_qubit_density = subfigs[0].subplots(4, 1, sharex=True)
    qubits_to_plot = [1, 3, 6, 9]
    plot_qubit_density(axs_qubit_density, qubits_to_plot, emu_sv_results, pulser_results)

    # Density difference plots
    axs_differences = subfigs[1].subplots(4, 1, sharex=True)
    plot_density_differences(
        axs_differences, qubits_to_plot, emu_sv_results, pulser_results
    )

    # Save the plot
    plt.savefig(output_dir / f"{output_dir.parent.name}.png")
