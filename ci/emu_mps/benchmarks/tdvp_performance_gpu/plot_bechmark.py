from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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


# to numpy
N = np.array([n**2 for n in N])
bond_dims = np.array(bond_dims)

scale_factor = 1.1
figsize = (scale_factor * 4, scale_factor * 3)
dpi = 300

fig, ax = plt.subplots(figsize=figsize)
for i, chi in enumerate(bond_dims):  # /chi**2
    ax.scatter(N, RSS[i, :] / chi**2, label=rf"$\chi$={chi}")
ax.plot(
    np.arange(10, 100, 1),
    worstRSS(1, np.arange(10, 100, 1), 30),
    color="black",
    label=r"$m(N,\chi,k=30)/\chi^2$",
)
ax.legend()
ax.set_xlabel("N")
ax.set_ylabel(r"RSS/ $\chi^2$ [GB]")
plt.tight_layout()
plt.savefig(script_dir / "RSS_vs_N.png")

# RUNTIME

# runtime 2D fit
X, Y = np.meshgrid(N, bond_dims)


def func(xy, b, c):
    x, y = xy
    # return a + b * x + c * y + d * x**2 + e * y**2 + f * x * y
    return b * (x**2) * (y**3) + c * (x**3) * (y**2)


# Perform curve fitting
popt, pcov = curve_fit(func, (X.flatten(), Y.flatten()), runtime.flatten())


fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
n_fit = np.arange(20, 100, 1)
for i, chi in enumerate(bond_dims):
    ax.scatter(N, runtime[i, :], label=rf"$\chi$={chi}")
    ax.plot(n_fit, func([n_fit, chi], *popt), label="fit")
ax.legend()
ax.set_xlabel("N")
ax.set_ylabel(r"$\langle\Delta t_{TDVP}\rangle$  [s]")
plt.tight_layout()
plt.savefig(script_dir / "runtime_vs_N.png")

fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
chi_fit = np.arange(200, 1000, 10)
for i, n in enumerate(N):
    ax.scatter(bond_dims, runtime[:, i], label=f"N={n}")
    ax.plot(chi_fit, func([n, chi_fit], *popt), label="fit")
ax.legend()
ax.set_xlabel(r"$\chi$")
ax.set_ylabel(r"$\langle\Delta t_{TDVP}\rangle$")
plt.tight_layout()
plt.savefig(script_dir / "runtime_vs_bond_dim.png")

print(*popt)
print(pcov)
print("perr:", np.sqrt(np.diag(pcov)))

""" # Create 3D plot of the data points and the fitted curve
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X, Y, runtime, color="blue")
runtime_fit = func((X, Y), *popt)
ax.plot_surface(X, Y, runtime_fit, color="red", alpha=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.savefig(script_dir / "prova_fit.png") """


delta = 1
chi = np.arange(0, 2050, delta)
n = np.arange(0, 205, delta)
N, CHI = np.meshgrid(n, chi)

runtime_estimate = func((N, CHI), *popt)

# plot
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

levels = [0, 1, 5, 10, 50, 100, 500, 1000]
CS = ax.contourf(N, CHI, runtime_estimate, cmap="viridis", levels=levels)
CS2 = ax.contour(N, CHI, runtime_estimate, colors="k", linewidths=0.5, levels=levels)
fig.colorbar(CS, ax=ax, label=r"$\langle\Delta t\rangle$ [s]")
ax.set_title("EMU-MPS")
ax.set_xlabel("N")
ax.set_ylabel(r"$\chi$")

plt.tight_layout()
plt.savefig(script_dir / "emumps_runtime_map.png")
