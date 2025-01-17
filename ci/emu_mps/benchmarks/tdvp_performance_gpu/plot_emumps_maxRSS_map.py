from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

script_dir = Path(__file__).parent


def rss_estimate(chi, n, max_krylov_dim):
    # worst case max RSS emu-mps
    return 4 * (chi**2) * (n**2 + 34 * n + 16 * max_krylov_dim + 64) / 1e9


# reasonable estimate of the max number of Krylov vectors
max_krylov_dim = 30
# meshgrid for contour plot
delta = 1
chi = np.arange(0, 2050, delta)
n = np.arange(0, 205, delta)
N, CHI = np.meshgrid(n, chi)

RSS = rss_estimate(CHI, N, max_krylov_dim)

# plot
scale_factor = 1.1
figsize = (scale_factor * 4, scale_factor * 3)
dpi = 300
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

levels = [0, 1, 5, 10, 20, 30, 40]
CS = ax.contourf(N, CHI, RSS, cmap="viridis", levels=levels)
CS2 = ax.contour(N, CHI, RSS, colors="k", levels=levels, linewidths=0.5)
ax.contour(N, CHI, RSS, colors="red", levels=[40], linewidths=0.8)
fig.colorbar(CS, ax=ax, label="max RSS [GB]")
ax.set_title("EMU-MPS memory footprint (k=30)", fontsize=10)
ax.set_xlabel("N")
ax.set_ylabel(r"$\chi$")

plt.tight_layout()
plt.savefig(script_dir / "emumps_maxRSS_map.png")
