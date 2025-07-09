# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm

import utils as u


# %%
# Data generation
# for each label a cluster center is generated
K = 10
d = 6

cov = np.eye(d)
sig_clust = 0
sig_noise = 3

rng = np.random.default_rng(14)

clt_center = rng.multivariate_normal(np.zeros(d), cov, K)


# %%

lbd_ub = 3 / K
p = 1 / (1 + lbd_ub * np.arange(K))
p = p / np.sum(p)

n_lrn = 2000  # size sample for the learning phase

X_train, y_train = u.sample_points(n_lrn, clt_center, sig_clust, sig_noise, p=p)

# %%
# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)

# Convert y label to float tensors
y_train = torch.FloatTensor(y_train).type(torch.LongTensor)


# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    def __init__(self, in_features=d, h1=15, h2=15, h3=15, out_features=K):
        super().__init__()  # instantiate our nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=-1)
        return x


# Pick a manual seed for randomization
torch.manual_seed(41)
# Create an instance of model
model = Model()

# Set the criterion of model to measure the error, here the negative log likelihood
criterion = nn.CrossEntropyLoss()  # nn.NLLLoss()
# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# %%
epochs = 1000
losses = []
for i in range(epochs):
    # Go forward and get a prediction
    y_pred = model.forward(X_train)  # Get predicted results

    # Measure the loss/error, gonna be high at first
    loss = criterion(y_pred, y_train)  # predicted values vs the y_train

    # Keep Track of our losses
    losses.append(loss.detach().numpy())

    # Do some back propagation: take the error rate of forward propagation and feed it back
    # thru the network to fine tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#########################################Conformal part########################################
# %%
alpha = 0.1
Nrepet = 5000
m = 100
Ncalscore = 100
method = [
    "MAJ",
    "BIN",
    "BetaBINcd",
    "MAJex",
]
methodscore = [
    "L1",
    "L2",
    "L2Id",
]
method = method + methodscore

Nmet = len(method)
lengths = np.zeros((Nmet, m, Nrepet))
ind_truelabel = np.zeros((Nmet, m, Nrepet))
ncal = 1000
score = "THR"
compQ = False  # indicate True if the Binomial Quantile or BetaBinomial quantile are not pre computed
QBin = None
Qnhgeo = None

randomize = True

load = True
if load:
    with open("Bin_quantile_lvl" + str(alpha) + "_m" + str(100), "rb") as fp:
        Qbin = pickle.load(fp)
    with open(
        "nHgeo_quantile_lvl" + str(alpha) + "_m" + str(100) + "n" + str(1000), "rb"
    ) as fp:
        Qnhgeo = pickle.load(fp)
else:
    Qbin = u.compute_binomial_quantile(m, alpha)
    Qnhgeo = u.compute_betabinomial_quantile(m, ncal, alpha)


# %% Computation of the sets for all the test points and all m
Xnew, ynew = u.sample_combination_points(
    Nrepet, m, clt_center, sig_clust, sig_noise, p=p
)
for j in tqdm(range(Nrepet)):
    Xcal, ycal = u.sample_points(
        ncal,
        clt_center,
        sig_clust,
        sig_noise,
        p=p,
    )
    Xcb, ycb = Xnew[j], ynew[j]
    yeval = u.prediction_np(Xcb, model)
    ycal_eval = u.prediction_np(Xcal, model)
    S_cal, S_new = u.compute_scores(ycal_eval, ycal, yeval, score)
    class_size = u.class_sizes(ycal)
    # compute p value and CP sets once for all methods and if conditional
    pvalcdrand = u.p_value(S_cal, ycal, S_new, cond=True, randomize=True)
    CPcd = u.conformal_set(S_cal, ycal, S_new, alpha, cond=True)
    CPMAJ = u.conformal_set(S_cal, ycal, S_new, alpha / 2, cond=True)
    pval_score_cal, ypval_score_cal = u.p_value_cal_score(class_size, m, Ncalscore)
    for imet, met in enumerate(method):
        if met[:3] == "MAJ":
            CPaux = CPMAJ
        else:
            CPaux = CPcd
        if met in methodscore:
            cp_set = u.combination_pvalscore(
                pvalcdrand, pval_score_cal, ypval_score_cal, alpha, met
            )
        else:
            cp_set = u.combination_majority_vote(
                CPaux,
                met,
                alpha,
                class_size=class_size,
                compQ=compQ,
                Qbin=Qbin,
                Qnhgeo=Qnhgeo,
            )
        lengths[imet, :, j] = u.length(cp_set)
        ind_truelabel[imet, :, j] = cp_set[:, ycb]

coverage = np.mean(ind_truelabel, axis=-1)
avg_lgths = np.mean(lengths, axis=-1)
coverage_cd = u.cond_average(ind_truelabel, ynew, K=K)
lgths_cd = u.cond_average(lengths, ynew, K=K)

# %% PLOTS
markers = ["o", "s", "v", "^", "D", "P", "*", "X", "<", ">"]
marker_size = 5
fontaxis = 12
cmap = sns.color_palette("colorblind", K)

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
# %%
coverage_std = np.sqrt(coverage * (1 - coverage) / 5000)
length_std = np.std(lengths, axis=-1)
coverage_cd_std = np.sqrt(np.max((1 - coverage_cd) * coverage_cd / p, axis=-1) / 5000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=False, sharey=False)
selected_method = method
m = 100

x = np.arange(1, m + 1)

# === Plot 1: Coverage ===
ax = axes[0]
for i, met in enumerate(method):
    if met in selected_method:
        ax.plot(
            x,
            coverage[i],
            c=cmap[i],
            marker=markers[i % len(markers)],
            markevery=10,
            markersize=marker_size,
        )
        ax.fill_between(
            x,
            coverage[i] - coverage_std[i],
            coverage[i] + coverage_std[i],
            color=cmap[i],
            alpha=0.2,
        )
ax.axhline(1 - alpha, c="k", linestyle="--")
ax.set_title("Coverage")
ax.set_xlim(1, m)
ax.set_xlabel("Number of observations")
ax.set_ylabel("Marginal coverage")
ax.grid(alpha=0.5)

# === Plot 2: Average length ===
ax = axes[1]
for i, met in enumerate(method):
    if met in selected_method:
        ax.plot(
            x,
            avg_lgths[i],
            c=cmap[i],
            marker=markers[i % len(markers)],
            markevery=10,
            markersize=marker_size,
        )
        ax.fill_between(
            x,
            avg_lgths[i] - length_std[i],
            avg_lgths[i] + length_std[i],
            color=cmap[i],
            alpha=0.1,
        )
ax.set_title("Average length")
ax.set_xlabel("Number of observations")
ax.set_ylabel("Size of the set")
ax.set_xlim(1, m)
ax.grid(alpha=0.5)

# === Plot 3: Avg conditional coverage ===
avg_cov_cd = np.min(coverage_cd, axis=-1)
ax = axes[2]
for i, met in enumerate(method):
    if met in selected_method:
        ax.plot(
            x,
            avg_cov_cd[i],
            c=cmap[i],
            marker=markers[i % len(markers)],
            markevery=10,
            markersize=marker_size,
        )
        ax.fill_between(
            x,
            avg_cov_cd[i] - coverage_cd_std[i],
            avg_cov_cd[i] + coverage_cd_std[i],
            color=cmap[i],
            alpha=0.2,
        )
ax.axhline(1 - alpha, c="k", linestyle="--")
ax.set_title("Worst Conditional Coverage")
ax.set_xlim(1, m)
ax.set_xlabel("Number of observations")
ax.set_ylabel("Worst Conditional Coverage")
ax.grid(alpha=0.5)

# === Légende commune ===
method_name = [
    "Majority",
    "Binomial",
    "Beta-Binomial",
    "Exchangeable majority",
    "Wilcoxon",
    r"$\ell_2$ Area",
    r"$\ell_2$",
]
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
    handles=handles,
    loc="lower center",
    ncol=len(handles),
    bbox_to_anchor=(0.5, -0.05),
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# %% distribution of the scores

m = 100
Nrepet = 5000
Xnew, ynew = u.sample_combination_points(
    Nrepet,
    m,
    clt_center,
    sig_clust,
    sig_noise,
    p=p,
)
Xcal, ycal = u.sample_points(
    Nrepet,
    clt_center,
    sig_clust,
    sig_noise,
    p=p,
)

yeval = u.prediction_np(Xnew, model)
ycal_eval = u.prediction_np(Xcal, model)


plt.figure(figsize=(6, 5))

S_cal = 1 - np.array([ycal_eval[i, y] for i, y in enumerate(ycal)])
S_new = 1 - np.array([yeval[i, :, y] for i, y in enumerate(ynew)])

for idx, m_val in enumerate(np.arange(10)[::2]):
    grey_shade = str(0.3 + 0.1 * idx)  # Nuances de gris entre 0.3 et 0.8
    plt.plot(
        np.sort(np.mean(S_new[:, : (m_val + 1)], axis=-1)),
        color=grey_shade,
        label=f"m={m_val+1}",
    )

plt.plot(np.sort(S_cal), color="black", linewidth=2, label="calibration")

plt.xticks([])
plt.ylabel("Score S")
plt.title("Distributions of the aggregated scores")
plt.grid()

# Légende en dessous
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0),
    ncol=3,
    fontsize=14,
)
plt.tight_layout()
plt.show()


# %% Naive methods results
def group_mean_scores(S, y, m):
    unique_labels = np.unique(y)
    grouped_means = []
    grouped_labels = []

    for label in unique_labels:
        S_label = S[y == label]
        n_full_groups = len(S_label) // m
        S_trimmed = S_label[: n_full_groups * m]

        means = S_trimmed.reshape(-1, m).mean(axis=1)

        grouped_means.append(means)
        grouped_labels.append(np.full(len(means), label))

    return np.concatenate(grouped_means), np.concatenate(grouped_labels)


# Parameters
num_runs = 1000
num_m = 100
Nrepet = 300
Ncal = 5000

# Initialization
cov_all = np.zeros((num_runs, num_m))
length_all = np.zeros((num_runs, num_m))
covmean_all = np.zeros((num_runs, num_m))
lengthmean_all = np.zeros((num_runs, num_m))

#
for run in tqdm(range(num_runs)):
    Xnew, ynew = u.sample_combination_points(
        Nrepet,
        m,
        clt_center,
        sig_clust,
        sig_noise,
        p=p,
    )
    Xcal, ycal = u.sample_points(
        Ncal,
        clt_center,
        sig_clust,
        sig_noise,
        p=p,
    )
    yeval = u.prediction_np(Xnew, model)
    ycal_eval = u.prediction_np(Xcal, model)
    S_cal = np.array([ycal_eval[i, y] for i, y in enumerate(ycal)])

    for mm in range(num_m):
        S_cal_mean, ymean = group_mean_scores(S_cal, ycal, mm + 1)

        # Method 2
        CPmean = u.conformal_set(
            S_cal_mean,
            ymean,
            np.mean(yeval[:, : (mm + 1)], axis=1),
            alpha=0.1,
            cond=True,
        )
        covmean_all[run, mm] = u.coverage(CPmean, ynew)
        lengthmean_all[run, mm] = np.mean(u.length(CPmean))

        # Method 1
        CP = u.conformal_set(
            S_cal, ycal, np.mean(yeval[:, : (mm + 1)], axis=1), alpha=0.1, cond=True
        )
        cov_all[run, mm] = u.coverage(CP, ynew)
        length_all[run, mm] = np.mean(u.length(CP))

# %% Means and std
cov = cov_all.mean(axis=0)
cov_std = cov_all.std(axis=0)
covmean = covmean_all.mean(axis=0)
covmean_std = covmean_all.std(axis=0)

length = length_all.mean(axis=0)
length_std = length_all.std(axis=0)
lengthmean = lengthmean_all.mean(axis=0)
lengthmean_std = lengthmean_all.std(axis=0)

# %% Plots
cov_std = np.sqrt(cov * (1 - cov) / (num_runs))
covmean_std = np.sqrt(covmean * (1 - covmean) / (num_runs))


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(1, num_m + 1)

# Plot coverage
(line1,) = axes[0].plot(x, cov, label="Direct calibration", color="tab:blue")
axes[0].fill_between(
    x, cov - cov_std, np.minimum(cov + cov_std, 1), color="tab:blue", alpha=0.2
)

(line2,) = axes[0].plot(x, covmean, label="Mean calibration", color="tab:orange")
axes[0].fill_between(
    x,
    covmean - covmean_std,
    np.minimum(covmean + covmean_std, 1),
    color="tab:orange",
    alpha=0.2,
)

axes[0].axhline(0.9, color="black", linestyle="--", linewidth=1)
axes[0].set_title("Marginal coverage")
axes[0].set_xlabel("Number of observations")
axes[0].set_ylabel("Coverage")
axes[0].grid(True)

# Plot longueur
axes[1].plot(x, length, color="tab:blue")
axes[1].fill_between(
    x, length - length_std, length + length_std, color="tab:blue", alpha=0.2
)

axes[1].plot(x, lengthmean, color="tab:orange")
axes[1].fill_between(
    x,
    lengthmean - lengthmean_std,
    lengthmean + lengthmean_std,
    color="tab:orange",
    alpha=0.2,
)

axes[1].set_title("Average length")
axes[1].set_xlabel("Number of observations")
axes[1].set_ylabel("Mean set size")
axes[1].grid(True)

legend = fig.legend(
    handles=[line1, line2],
    labels=["Direct calibration", "Mean calibration"],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=2,
    fontsize=16,
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.subplots_adjust(bottom=0.2)
plt.show()

# %%
