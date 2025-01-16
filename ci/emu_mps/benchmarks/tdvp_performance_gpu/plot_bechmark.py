from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# from scipy.optimize import curve_fit

script_dir = Path(__file__).parent
res_dir = script_dir / "results"
res_dir.mkdir(exist_ok=True)

title = "TDVP performance - GPU"


def listofdict_to_dictoflist(listofdict: list[dict]):
    return {key: [d[key] for d in listofdict] for key in listofdict[0].keys()}


# import results
N = [9, 8, 7, 6]
bond_dims = [900, 800, 700, 600, 500]
shape = (len(bond_dims), len(N))
runtime = np.zeros(shape)
RSS = np.zeros(shape)

for i, n in enumerate(N):
    for j, dim in enumerate(bond_dims):
        with open(res_dir / f"results_{n}_{dim}.json", "r") as file:
            res_dict = json.load(file)
            res = listofdict_to_dictoflist(res_dict["statistics"]["steps"])
            RSS[j, i] = max(res["RSS"]) / 1e3
            runtime[j, i] = sum(res["duration"][10:]) / len(res["duration"][10:])


max_krylov_dim = 20


def worstRSS(chi, n, krylov_dim):
    return 4 * (chi**2) * (n**2 + 34 * n + 16 * krylov_dim + 64) / 1e9


scale_factor = 1.1
figsize = (scale_factor * 4, scale_factor * 3)
dpi = 300

fig, ax = plt.subplots(figsize=figsize)
for i, chi in enumerate(bond_dims):  # /chi**2
    ax.scatter(np.array(N) ** 2, RSS[i, :] / chi**2, label=rf"$\chi$={chi}")
ax.plot(
    np.arange(10, 100, 1),
    worstRSS(1, np.arange(10, 100, 1), 30),
    color="black",
    label=r"$maxRSS(N,k=30)$",
)
ax.legend()
ax.set_xlabel("N")
ax.set_ylabel(r"RSS/ $\chi^2$ [GB]")
plt.tight_layout()
plt.savefig(script_dir / "RSS_vs_N.png")

fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
for i, n in enumerate(N):
    ax.scatter(bond_dims, RSS[:, i], label=f"N = {n**2}")
    ax.plot(
        np.arange(400, 1000, 1),
        worstRSS(np.arange(400, 1000, 1), n**2, 30),
        color="black",
        label=r"$maxRSS(N,k=30)$",
    )
ax.legend()
ax.set_xlabel(r"$\chi$")
ax.set_ylabel("RSS [GB]")
plt.tight_layout()
plt.savefig(script_dir / "RSS_vs_bond_dim.png")

fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
for i, chi in enumerate(bond_dims):  # / chi**3
    ax.plot(np.array(N) ** 2, runtime[i, :] / chi**3, label=rf"$\chi$={chi}")
ax.legend()
ax.set_xlabel("N")
ax.set_ylabel(r"$\langle\Delta t_{TDVP}\rangle$  [s]")
plt.tight_layout()
plt.savefig(script_dir / "runtime_vs_N.png")

fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
for i, n in enumerate(N):  # / n**2
    ax.plot(bond_dims, runtime[:, i] / n**4, label=f"N = {n**2}")
ax.legend()
ax.set_xlabel(r"$\chi$")
ax.set_ylabel(r"$\langle\Delta t_{TDVP}\rangle$")
plt.tight_layout()
plt.savefig(script_dir / "runtime_vs_bond_dim.png")
