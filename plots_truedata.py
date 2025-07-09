# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
temp = 1
alpha = 0.1
minclasssize = 20
nameload = (
    "Plantnet_shuffle_temp"
    + str(temp)
    + "alpha"
    + str(alpha)
    + "minclasssize"
    + str(minclasssize)
    + ".npy"
)
Data1 = np.load(nameload, allow_pickle=True)
coveragetp1 = Data1[()]["coverage"]
avg_lgthstp1 = Data1[()]["avg_lgths"]
coverage_cdtp1 = Data1[()]["coverage_cd"]
lgths_cdtp1 = Data1[()]["lgths_cd"]
method = Data1[()]["methods"]


temp = 20
alpha = 0.1
minclasssize = 20
nameload = (
    "Plantnet_shuffle_temp"
    + str(temp)
    + "alpha"
    + str(alpha)
    + "minclasssize"
    + str(minclasssize)
    + ".npy"
)
Data20 = np.load(nameload, allow_pickle=True)
coveragetp20 = Data20[()]["coverage"]
avg_lgthstp20 = Data20[()]["avg_lgths"]
coverage_cdtp20 = Data20[()]["coverage_cd"]
lgths_cdtp20 = Data20[()]["lgths_cd"]
# methodtp20 = Data20[()]["methods"]


avg_cov_cdtp1 = np.nanmean(coverage_cdtp1, axis=-1).T
avg_lgths_cdtp1 = np.nanmean(lgths_cdtp1, axis=-1).T


avg_cov_cdtp20 = np.nanmean(coverage_cdtp20, axis=-1).T
avg_lgths_cdtp20 = np.nanmean(lgths_cdtp20, axis=-1).T

maxsizeobs = 10

cmap = sns.color_palette("colorblind", 30)
markers = ["o", "s", "v", "^", "D", "P", "*", "X", "<", ">"]

colormethod = {met: cmap[i] for i, met in enumerate(method)}
markermethod = {met: markers[i] for i, met in enumerate(method)}


plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

plt.rcParams.update(
    {
        "font.size": 16,  # base font size
        "axes.titlesize": 18,  # subplot titles
        "axes.labelsize": 16,  # x/y labels
        "legend.fontsize": 16,  # legend text
        "xtick.labelsize": 16,  # tick labels
        "ytick.labelsize": 16,
    }
)
markers = ["o", "s", "v", "^", "D", "P", "*", "X", "<", ">"]
marker_size = 5

fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharey="row")


# === Length tp1
for i, met in enumerate(method):
    axs[0].plot(
        np.arange(1, maxsizeobs + 1),
        avg_lgthstp1[i],
        c=colormethod[met],
        marker=markermethod[met],
        markersize=marker_size,
    )
axs[0].set_title("Average Length (Temp$=1$)")
axs[0].set_ylabel("Size of the set")
axs[0].set_xlabel("Number of observations")
axs[0].grid(alpha=0.5)


# === Length tp20
for i, met in enumerate(method):
    axs[1].plot(
        np.arange(1, maxsizeobs + 1),
        avg_lgthstp20[i],
        c=colormethod[met],
        marker=markermethod[met],
        markersize=marker_size,
    )
axs[1].set_title("Average Length (Temp$=20$)")
axs[1].set_xlabel("Number of observations")
axs[1].grid(alpha=0.5)

methodname_dict = {
    "MAJcd": "Majority",
    "MAJexcd": "Exch. majority",
    "MAJ": "Majority",
    "MAJex": "Exch. majority",
    "BIN": "Binomial",
    "BetaBINcd": "Beta-Binomial",
    "L1": "Wilcoxon",
    "L2": r"$\ell_2$ Area",
    "L2Id": r"$\ell_2$",
    "ScEnv": "Quantile",
}
method_name = [methodname_dict[met] for met in method]


handles = [
    plt.Line2D(
        [0],
        [0],
        color=colormethod[met],
        label=method_name[i],
        marker=markermethod[met],
        markevery=10,
        markersize=marker_size,
    )
    for i, met in enumerate(method)
]
legend = fig.legend(
    handles,
    method_name,
    loc="lower center",
    ncol=len(method_name) / 2,
    bbox_to_anchor=(0.5, -0.1),
    fontsize=12,
)

plt.subplots_adjust(wspace=0.1)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# %% Plot coverage
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey="row")

# === Coverage tp1
for i, met in enumerate(method):
    axs[0, 0].plot(
        np.arange(1, maxsizeobs + 1),
        coveragetp1[i],
        c=cmap[i],
        marker=markers[i],
        markersize=marker_size,
    )
axs[0, 0].axhline(1 - alpha, c="k", linestyle="--")
axs[0, 0].set_title("Coverage ($Temp=1$)")
axs[0, 0].set_ylabel("Marginal coverage")
axs[0, 0].grid(alpha=0.5)

# === Coverage tp20
for i, met in enumerate(method):
    axs[0, 1].plot(
        np.arange(1, maxsizeobs + 1),
        coveragetp20[i],
        c=cmap[i],
        marker=markers[i],
        markersize=marker_size,
    )
axs[0, 1].axhline(1 - alpha, c="k", linestyle="--")
axs[0, 1].set_title("Coverage ($Temp=20$)")
axs[0, 1].grid(alpha=0.5)

# === Average conditional coverage tp1
for i, met in enumerate(method):
    axs[1, 0].plot(
        np.arange(1, maxsizeobs + 1),
        avg_cov_cdtp1[i],
        c=cmap[i],
        marker=markers[i],
        markersize=marker_size,
    )
axs[1, 0].axhline(1 - alpha, c="k", linestyle="--")
axs[1, 0].set_title("Average Conditional Coverage ($Temp=1$)")
axs[1, 0].set_ylabel("Avg. Cond. Coverage")
axs[1, 0].set_xlabel("Number of observations")
axs[1, 0].grid(alpha=0.5)

# === Average conditional coverage tp20
for i, met in enumerate(method):
    axs[1, 1].plot(
        np.arange(1, maxsizeobs + 1),
        avg_cov_cdtp20[i],
        c=cmap[i],
        marker=markers[i],
        markersize=marker_size,
    )
axs[1, 1].axhline(1 - alpha, c="k", linestyle="--")
axs[1, 1].set_title("Average Conditional Coverage ($Temp=20$)")
axs[1, 1].set_xlabel("Number of observations")
axs[1, 1].grid(alpha=0.5)


handles = [
    plt.Line2D(
        [0],
        [0],
        color=cmap[i],
        label=method_name[i],
        marker=markers[i % len(markers)],
        markevery=10,
        markersize=marker_size,
    )
    for i, _ in enumerate(method)
]
fig.legend(
    handles,
    method_name,
    loc="lower center",
    ncol=len(method_name),
    bbox_to_anchor=(0.5, 0.02),
    fontsize=12,
)

plt.subplots_adjust(hspace=0.3, wspace=0.1)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()

# %% Plot results for true obsveration structure
temp = 1
alpha = 0.1
minclasssize = 20
nameload = (
    "Plantnet_trueobs_temp"
    + str(temp)
    + "alpha"
    + str(alpha)
    + "minclasssize"
    + str(minclasssize)
    + ".npy"
)
DataTrue = np.load(nameload, allow_pickle=True)
coverage = DataTrue[()]["coverage"]
avg_lgths = DataTrue[()]["avg_lgths"]
coverage_cd = DataTrue[()]["coverage_cd"]
lgths_cd = DataTrue[()]["lgths_cd"]
method = DataTrue[()]["methods"]


marker_size = 7

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# === Coverage tp1
for i, met in enumerate(method):
    axs[0].plot(
        np.arange(1, maxsizeobs + 1),
        coverage[i],
        c=cmap[i],
        marker=markers[i],
        markersize=marker_size,
    )
axs[0].axhline(1 - alpha, c="k", linestyle="--")
axs[0].set_title("Coverage")
axs[0].set_ylabel("Marginal coverage")
axs[0].set_xlabel("Number of observations")
axs[0].grid(alpha=0.5)


# === Length tp20
for i, met in enumerate(method):
    axs[1].plot(
        np.arange(1, maxsizeobs + 1),
        avg_lgths[i],
        c=cmap[i],
        marker=markers[i],
        markersize=marker_size,
    )
axs[1].set_title("Average Length")
axs[1].set_xlabel("Number of observations")
axs[1].grid(alpha=0.5)

method_name = [methodname_dict[met] for met in method]

handles = [
    plt.Line2D(
        [0],
        [0],
        color=cmap[i],
        label=method_name[i],
        marker=markers[i % len(markers)],
        markevery=10,
        markersize=marker_size,
    )
    for i, _ in enumerate(method)
]
fig.legend(
    handles,
    method_name,
    loc="lower center",
    ncol=len(method_name),
    bbox_to_anchor=(0.5, 0.02),
    fontsize=11,
)

plt.subplots_adjust(hspace=0.3, wspace=0.1)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
# %%
