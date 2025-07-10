# %%
from lxml import etree
import numpy as np
import os
import pickle

ntrain = 113204  # number of images

# %% load labels
y = []
obsID = []

# path may need to be adapted and put a default value of /home/jsalmon/Data/PlantCLEF2015/

default_dataset_path = "/home/jean-baptiste/Documents/Github/PlantNet-300K"
user_input = input(
    f"Please provide the path to the dataset directory [default: {default_dataset_path}]: "
).strip()
dataset_path = user_input if user_input else default_dataset_path


for n in range(ntrain):
    path_name_train = os.path.join(
        dataset_path, "PlantCLEF2015TrainingData", "train", str(n) + ".xml"
    )
    # "PlantCLEF2015TrainingData/train/" + str(n) + ".xml"
    path_name_test = os.path.join(
        dataset_path, "PlantCLEF2015TestDataWithAnnotations", str(n) + ".xml"
    )
    image_exists = False
    if os.path.exists(path_name_train):
        tree = etree.parse(path_name_train)
        image_exists = True
    elif os.path.exists(path_name_test):
        tree = etree.parse(path_name_test)
        image_exist = True
    if image_exists:
        root = tree.getroot()
        y += [int(root.findall("ClassId")[0].text)]
        obsID += [int(root.findall("ObservationId")[0].text)]

y = np.array(y)
obsID = np.array(obsID)


# %% Re-numbering of the labels and creating array of obsId
trueclasses, newy = np.unique(y, return_inverse=True)
trueobs, newObsID = np.unique(obsID, return_inverse=True)

nobs = trueobs.shape[0]

np.save("newlabels", newy)
np.save("Images_to_Obs", newObsID)

# %% Create Obs_to_Images file
Obs_to_Images = [[] for _ in range(nobs)]
for ind, obs in enumerate(newObsID):
    Obs_to_Images[obs] += [ind]

with open("Obs_to_Images", "wb") as fp:  # Pickling
    pickle.dump(Obs_to_Images, fp)


# %% split data between train-cal-test,
# observations with numerous images are preferably attributed to test set
# this original split may break the exchangeability between calibration and test, to avoid this the calibration and test can be reshuffled in CP_Truedata.py
train_ind = []
test_ind = []
cal_ind = []
ntrain = 0
ntest = 0
ncal = 0

nobs = np.unique(newObsID).shape[0]
obs_size = np.zeros(nobs)
y_obs = np.zeros(nobs)
for obs in range(nobs):
    obs_size[obs] = len(Obs_to_Images[obs])
    y_obs[obs] = newy[Obs_to_Images[obs][0]]

for label in range(1000):
    # range(len(np.unique(y_obs))):
    obss = np.where(y_obs == label)[0]
    size_obss = obs_size[obss]
    nytr = 0
    nyts = 0
    nycal = 0
    if len(obss) > 2:
        obss_avail = np.array([True for _ in obss])
        for _ in range(len(obss)):
            if nyts <= nytr and nyts <= nycal:
                obsind = np.argmax(size_obss[obss_avail])
                test_ind += Obs_to_Images[obss[obss_avail][obsind]]
                nyts += size_obss[obss_avail][obsind]
            elif nytr <= nycal:
                obsind = np.argmin(size_obss[obss_avail])
                train_ind += Obs_to_Images[obss[obss_avail][obsind]]
                nytr += size_obss[obss_avail][obsind]
            else:
                obsind = np.argmin(size_obss[obss_avail])
                cal_ind += Obs_to_Images[obss[obss_avail][obsind]]
                nycal += size_obss[obss_avail][obsind]
            aux = obss_avail[obss_avail]
            aux[obsind] = False
            obss_avail[obss_avail] = aux
    else:
        obsind = np.argmin(size_obss)
        if size_obss[obsind] == 1:
            obsind = 1 - obsind
        test_ind += Obs_to_Images[obss[1 - obsind]]
        minsize = int(size_obss[obsind])
        cal_ind += Obs_to_Images[obss[obsind]][: minsize // 2]
        train_ind += Obs_to_Images[obss[obsind]][minsize // 2 :]

np.save("train_ind", train_ind)
np.save("test_ind", test_ind)
np.save("cal_ind", cal_ind)

# %%
