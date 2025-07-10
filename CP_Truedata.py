# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sklearn as sk
import seaborn as sns
import pickle
import utils as u
import os
from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle
from tqdm import tqdm


# %% load all quantities
DataTest = np.load("DataTest.npy")
DataCal = np.load("DataCal.npy")


temp = 1
DataTest[:, 1:] = sc.special.softmax(DataTest[:, 1:] / temp, axis=-1)
DataCal[:, 1:] = sc.special.softmax(DataCal[:, 1:] / temp, axis=-1)

test_ind = DataTest[:, 0].astype(int)
cal_ind = DataCal[:, 0].astype(int)

labels = np.load("newlabels.npy")
obs_Id = np.load("Images_to_Obs.npy")
with open("Obs_to_Images", "rb") as fp:
    Obs_to_Images = pickle.load(fp)

obstest = np.unique(obs_Id[test_ind.astype(int)])
maxsizeobs = 10
K = 1000


# %% Clean Obs_to_Images with for images belonging to train_set
for obsind, obs in enumerate(Obs_to_Images):
    aux = obs.copy()
    newobs = []
    for image in aux:
        if (image in cal_ind) or (image in test_ind):
            newobs += [image]
    Obs_to_Images[obsind] = newobs


# %% Reshuffle calibration and test and keeping observation structure if wanted
mergeobs = False
if mergeobs:
    newtest = []
    newcal = []
    ntest = 0
    ncal = 0

    listeobs = np.unique(np.append(obs_Id[cal_ind], obs_Id[test_ind]))
    nobs = listeobs.shape[0]
    obs_size = np.zeros(nobs)
    y_obs = np.zeros(nobs)
    for obsind, obs in enumerate(listeobs):
        obs_size[obsind] = len(Obs_to_Images[obs])
        y_obs[obsind] = labels[Obs_to_Images[obs][0]]

    for label in range(K):
        idobss = np.where(y_obs == label)[0]
        obss = listeobs[idobss]
        obss = shuffle(obss)
        size_obss = obs_size[idobss]
        nyts = 0
        nycal = 0
        for ii in range(len(obss)):
            if nycal <= nyts:
                newcal += Obs_to_Images[obss[ii]]
                nycal += size_obss[ii]
            else:
                newtest += Obs_to_Images[obss[ii]]
                nyts += size_obss[ii]
        ntest += nyts
        ncal += nycal
    Data = np.append(DataCal, DataTest, axis=0)
    calsort = np.sort(newcal)
    dataargsort = np.argsort(
        Data[:, 0],
    )
    jj = 0
    selectedcal = np.zeros(Data.shape[0], dtype=bool)
    for ind in dataargsort:
        if Data[ind, 0].astype(int) == calsort[jj]:
            selectedcal[ind] = True
            jj += 1
        if jj == calsort.shape[0]:
            break
    shuffle_DataCal = Data[selectedcal]
    shuffle_DataTest = Data[(1 - selectedcal).astype(bool)]
    DataCal, DataTest = shuffle_DataCal, shuffle_DataTest
    cal_ind, test_ind = newcal, newtest


# %% Delete classes with few data in the calibration
minclasssize = 20  # minimum size of the class
class_size_cal = u.class_sizes(labels[cal_ind])
selected_class = class_size_cal > minclasssize
K = np.sum(selected_class)
boolcal_delete = [selected_class[labels[ind]] for ind in cal_ind]
booltest_delete = [selected_class[labels[ind]] for ind in test_ind]
column_kept = np.append(
    True, selected_class
)  # keep the first column which contains the Id
DataCal = DataCal[boolcal_delete][:, column_kept]
DataTest = DataTest[booltest_delete][:, column_kept]
test_ind = DataTest[:, 0].astype(int)
cal_ind = DataCal[:, 0].astype(int)

aux_newlabels = np.cumsum(selected_class) - 1  # give new labels to the kept classes
labels = np.array([aux_newlabels[y] for y in labels])
obstest = np.unique(obs_Id[test_ind.astype(int)])


# %% Create list of true obs
allobs_index_true = [[] for _ in range(maxsizeobs)]
allobs_label_true = [[] for _ in range(maxsizeobs)]
for obsind in obstest:
    images = Obs_to_Images[obsind]
    size = len(images)
    y = labels[images[0]]
    if size <= maxsizeobs:
        allobs_index_true[size - 1] += [images]
        allobs_label_true[size - 1] += [y]


# %% Shuffle test and calibration sets without keeping the observation structure
reshuffle = False
split = 0.5
if reshuffle:
    new_DataCal, new_DataTest = train_test_split(
        np.append(DataCal, DataTest, axis=0),
        train_size=split,
        stratify=np.append(
            labels[DataCal[:, 0].astype(int)], labels[DataTest[:, 0].astype(int)]
        ),
    )
    nwtest_ind = new_DataTest[:, 0].astype(int)
    nwcal_ind = new_DataCal[:, 0].astype(int)
    class_size_cal = u.class_sizes(labels[nwcal_ind])
    selected_class = class_size_cal > minclasssize
    K = np.sum(selected_class)
    boolcal_delete = [selected_class[labels[ind]] for ind in nwcal_ind]
    booltest_delete = [selected_class[labels[ind]] for ind in nwtest_ind]
    column_kept = np.append(
        True, selected_class
    )  # keep the first column which contains the Id
    new_DataCal = new_DataCal[boolcal_delete][:, column_kept]
    new_DataTest = new_DataTest[booltest_delete][:, column_kept]

    aux_newlabels = np.cumsum(selected_class) - 1  # give new number to the kept classes
    labels = np.array([aux_newlabels[y] for y in labels])

    allobs_index_shuffle = [[] for _ in range(maxsizeobs)]
    allobs_label_shuffle = [[] for _ in range(maxsizeobs)]
    for y in range(K):
        indy = np.intersect1d(new_DataTest[:, 0], np.where(labels == y))
        random.shuffle(indy)
        for m in range(maxsizeobs):
            for i in range(len(indy) // (m + 1)):
                allobs_index_shuffle[m] += [indy[i * (m + 1) : (i + 1) * (m + 1)]]
                allobs_label_shuffle[m] += [y]

    # subshuffle a number Nrepet of multi-input
    allobs_index_subshuffle = [[] for _ in range(maxsizeobs)]
    allobs_label_subshuffle = [[] for _ in range(maxsizeobs)]
    Nrepet = 3000
    for m in range(maxsizeobs):
        newlab, newind = shuffle(allobs_label_shuffle[m], allobs_index_shuffle[m])
        allobs_index_subshuffle[m] = newind[: min(len(newind), Nrepet)]
        allobs_label_subshuffle[m] = newlab[: min(len(newind), Nrepet)]


# %% Calculate score

alpha = 0.1
scoremet = True
Ncalscore = 1000
threshold_Pnet = 0.01

method = ["MAJcd", "MAJexcd", "BIN", "BetaBINcd"]
methodscore = [
    "L1",
    "L2",
    "L2Id",
    "ScEnv",
]
method = method + methodscore


Nmet = len(method)
lengths = []
score = "THR"
compQ = False
QBin = None
Qnhgeo = None

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


# %% Choose the type of observation structure
if reshuffle:
    DataCal, DataTest = new_DataCal, new_DataTest
    allobs_index = allobs_index_subshuffle
    allobs_label = allobs_label_subshuffle
else:
    allobs_index = allobs_index_true
    allobs_label = allobs_label_true
# %% calibration scores
ycal_eval = DataCal[:, 1:]
ycal = labels[DataCal[:, 0].astype(int)]
yeval = DataTest[:, 1:]
ynew_all = labels[DataTest[:, 0].astype(int)]

S_cal, S_new_all = u.compute_scores(ycal_eval, ycal, yeval, score)
class_size = u.class_sizes(ycal)
pvalcdrand_all = u.p_value(S_cal, ycal, S_new_all, cond=True, randomize=True)
CPcd_all = u.conformal_set(S_cal, ycal, S_new_all, alpha, cond=True)
CPcd_MAJ_all = u.conformal_set(S_cal, ycal, S_new_all, alpha / 2, cond=True)


# %% Vectorization of the observations
posDataTest = np.zeros(int(np.max(DataTest[:, 0]) + 1), dtype=int)
for i, id in enumerate(DataTest[:, 0]):
    posDataTest[int(id)] = i

# %% Computation of the sets
lengths = [np.zeros((len(method), len(listeobs))) for listeobs in allobs_index]
ind_truelabel = [np.zeros((len(method), len(listeobs))) for listeobs in allobs_index]
for m, listeobs in tqdm(enumerate(allobs_index)):
    listeobs = np.array(listeobs, dtype=int)
    pval_score_cal, ypval_score_cal = u.p_value_cal_score_fast(
        class_size, m + 1, Ncalscore
    )
    indices = posDataTest[listeobs]
    S_new = S_new_all[indices]
    ycb = allobs_label[m]
    pvalcdrand = pvalcdrand_all[indices]
    CPcd = CPcd_all[indices]
    CPMAJ = CPcd_MAJ_all[indices]
    for imet, met in enumerate(method):
        if met in methodscore:
            paux = pvalcdrand
        elif met in ["BIN", "BetaBINcd"]:
            CPaux = CPcd
        elif met[:3] == "MAJ":
            CPaux = CPMAJ

        if met in methodscore:
            cp_set = u.combination_pvalscore_Multobs(
                paux,
                pval_score_cal,
                ypval_score_cal,
                alpha,
                met,
                class_size=class_size,
                test=True,
            )
        else:
            cp_set = u.combination_majority_vote_Multobs(
                CPaux,
                met,
                alpha,
                class_size=class_size,
                compQ=compQ,
                Qbin=Qbin,
                Qnhgeo=Qnhgeo,
            )
        lengths[m][imet, :] = np.sum(cp_set, axis=1)
        for ii in range(len(listeobs)):
            ind_truelabel[m][imet, ii] = cp_set[ii, ycb[ii]]


coverage = [np.mean(ind, axis=-1) for ind in ind_truelabel]
avg_lgths = [np.mean(length, axis=-1) for length in lengths]

coverage_cd = [
    u.cond_average(ind, allobs_label[m], K=K) for m, ind in enumerate(ind_truelabel)
]
lgths_cd = [
    u.cond_average(length, allobs_label[m], K=K) for m, length in enumerate(lengths)
]

# %%
result = {
    "methods": method,
    "alpha": alpha,
    "coverage": coverage,
    "avg_lgths": avg_lgths,
    "coverage_cd": coverage_cd,
    "lgths_cd": lgths_cd,
    "threshold_Pnet": threshold_Pnet,
    "temp": temp,
    "class_size": class_size,
}

if reshuffle:
    dataname = "shuffle"
else:
    dataname = "trueobs"
name = (
    "Plantnet_"
    + dataname
    + "_temp"
    + str(temp)
    + "alpha"
    + str(alpha)
    + "minclasssize"
    + str(minclasssize)
)
np.save(name, result)
