from pathlib import Path
import re
import json
import time


def parse_latest_log(logfile: Path, output_name: str = None) -> dict:
    """
    Returns a dictionary of performance metrics of an Emu-TN run.
    Optionally dump it as a json file with the `name` kwarg.

    Note: log file creation by EmuTN can be slow.
    Wait for it in the while loop.
    """
    logdir = logfile.parent
    time_counter = 0
    while not logfile.exists():
        time.sleep(1)
        time_counter += 1
        if time_counter > 30:
            raise FileNotFoundError("EmuTN log file not found")

    # make the dictionary
    logres = {}
    # get the labels
    with open(str(logfile), "r") as f:
        lines = f.readlines()
        for line in lines:
            if "step = " in line:
                labels = re.findall(r"(\S+) = ", line)
                for key in labels:
                    logres[key] = []
                break

    # store parsed values
    with open(logfile, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "step = " in line:
                values = re.findall(r" = ([\d.]+)", line)
                for i, key in enumerate(logres.keys()):
                    logres[key].append(float(values[i]))

    if output_name:
        print("\t\tmax(χ):", max(logres["χ"]))
        print("\t\tmax(|ψ|):", max(logres["|ψ|"]), "MiB")
        print("\t\tmax(RSS):", max(logres["RSS"]), "MiB")
        runtime = sum(logres["Δt"])
        print(f"\t\truntime: {runtime:.3f} s")
        with open(logdir / (output_name + ".json"), "w") as file:
            json.dump(logres, file)

    # clean
    logfile.unlink()

    return logres


def parse_benckmark_results_2d(res_dir: Path, Nxs: list, Nys: list):
    max_res = {
        "N": [],
        "χ": [],
        "|ψ|": [],
        "RSS": [],
        "runtime": [],
    }

    for Nx in Nxs:
        for Ny in Nys:
            with open(res_dir / f"Nx{Nx}Ny{Ny}.json", "r") as file:
                res = json.load(file)
                max_res["N"].append(Nx * Ny)
                max_res["χ"].append(max(res["χ"]))
                max_res["|ψ|"].append(max(res["|ψ|"]))
                max_res["RSS"].append(max(res["RSS"]))
                max_res["runtime"].append(sum(res["Δt"]))
    # store biggest run for later plots
    max_res["maxNx"] = max(Nxs)
    max_res["maxNy"] = max(Nys)

    dictname = res_dir.parent.parent.name + ".json"
    with open(res_dir / dictname, "w") as file:
        json.dump(max_res, file)
