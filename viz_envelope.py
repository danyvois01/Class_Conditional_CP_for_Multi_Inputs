# %%
import pickle
import matplotlib.pyplot as plt
import utils as u
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# %% Generation envelope
# Parameters
alpha = 0.1
m = 10
n = 100
Nenv = 100000

sample_cal = u.gen_pvalues_vectorized(m, n, N_try=Nenv)
scores = u.Score_Quantile(sample_cal.T, n)
quant = np.sort(scores)[int(alpha * (Nenv + 1))]
envelope = u.gen_envelope_quant(n, m, quant)


# %% Plot
sample = u.gen_pvalues_vectorized(m, n, N_try=1000)

# Calculations
bonferonni = np.repeat(alpha / m, m)
simes = alpha * np.arange(1, m + 1) / m
gasparin = np.zeros(m)
gasparin[m // 2 :] = alpha * np.ones(m - m // 2)


# Plotting
fig, ax = plt.subplots(figsize=(6, 6))

# Plot data on main axes
ax.plot(np.arange(1, m + 1), envelope, label="Quantile", linewidth=2.5)
ax.plot(np.arange(1, m + 1), bonferonni, label="Bonferonni", linewidth=2.5)
ax.plot(np.arange(1, m + 1), simes, label="Simes", linewidth=2.5)
ax.plot(np.arange(1, m + 1), gasparin, label="Majority", linewidth=2.5)
ax.plot(np.arange(1, m + 1), sample[:150].T, color="k", alpha=0.15)

# Axis labels
ax.set_xlabel(r"$j$", fontsize=12)
ax.set_ylabel(r"$P_j$", fontsize=12)
ax.set_xticks(np.arange(1, m + 1))
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

ax.set_xlim(0.5, m + 0.5)
ax.set_ylim(-0.05, 1.05)

# Place legend inside the plot
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=True, fontsize=10
)

# Create inset axes
axins = zoomed_inset_axes(ax, zoom=2.5, loc="upper left")

# Plot data on inset axes
axins.plot(np.arange(1, m + 1), envelope, linewidth=2)
axins.plot(np.arange(1, m + 1), bonferonni, linewidth=2)
axins.plot(np.arange(1, m + 1), simes, linewidth=2)
axins.plot(np.arange(1, m + 1), gasparin, linewidth=2)
axins.plot(np.arange(1, m + 1), sample[:150].T, color="k", alpha=0.12)

# Set limits for inset axes
axins.set_xlim(1, 2.5)
axins.set_ylim(-0.02, 0.15)

# Hide ticks for cleaner look
axins.set_xticks([])
axins.set_yticks([])

# Adjust spines of inset axes
for axis in ["top", "bottom", "left", "right"]:
    axins.spines[axis].set_linewidth(3)
    axins.spines[axis].set_color("black")

# Mark the inset
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="black", lw=3, zorder=15)

# Use tight_layout to adjust the plot
plt.tight_layout()
plt.show()

# %%
