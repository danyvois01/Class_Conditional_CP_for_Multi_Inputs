# %%
from torchvision.models import resnet50
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import scipy as sc

# %%
# Load indices of test and cal
y = np.load("newlabels.npy")
test_ind = np.load("test_ind.npy")
cal_ind = np.load("cal_ind.npy")

# Image 1283 is missing in the dataset
cal_ind = np.delete(cal_ind, np.where(cal_ind == 1283)[0])


#  Definition of the Dataset class using to merge the two datasets
class DoubleDatasetTestCal(Dataset):
    def __init__(self, image_folder1, image_folder2, indices, labels, transform=None):
        self.image_folder1 = image_folder1
        self.image_folder2 = image_folder2
        self.indices = indices
        self.labels = labels
        self.transform = transform

        self.available_indices1 = [
            os.path.exists(os.path.join(image_folder1, f"{i}.jpg"))
            for i in range(len(labels))
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        if self.available_indices1[real_idx]:
            img_path = os.path.join(self.image_folder1, f"{real_idx}.jpg")
        else:
            img_path = os.path.join(self.image_folder2, f"{real_idx}.jpg")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, real_idx


# %% Define transformation of images
image_size = 256
crop_size = 224
transform_train = transforms.Compose(
    [
        transforms.Resize(size=image_size),
        transforms.RandomCrop(size=crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4425, 0.4695, 0.3266], std=[0.2353, 0.2219, 0.2325]
        ),
    ]
)
# %%
# Load parameters
model = resnet50(num_classes=1000)
model.load_state_dict(torch.load("model.pth"))
model.eval()


# %% Load data
image_folder1 = "/PlantCLEF2015TestDataWithAnnotations/"
image_folder2 = "/PlantCLEF2015TrainingData/"

datasettest = DoubleDatasetTestCal(
    image_folder1, image_folder2, test_ind, y, transform=transform_train
)
dataloadertest = DataLoader(datasettest, batch_size=32, shuffle=False, num_workers=4)

X = torch.zeros((len(datasettest), 1001))
total_test = 0

for input, id in dataloadertest:
    with torch.no_grad():
        pred = model(input)
        batchsize = id.size(0)
        X[total_test : total_test + batchsize, 0] = id
        X[total_test : total_test + batchsize, 1:] = pred
        total_test += batchsize
X = X.numpy()
np.save("DataTest", X)

datasetcal = DoubleDatasetTestCal(
    image_folder1, image_folder2, cal_ind, y, transform=transform_train
)
dataloadercal = DataLoader(datasetcal, batch_size=32, shuffle=False, num_workers=4)

X = torch.zeros((len(datasetcal), 1001))
total_test = 0

for input, id in dataloadercal:
    with torch.no_grad():
        pred = model(input)
        batchsize = id.size(0)
        X[total_test : total_test + batchsize, 0] = id
        X[total_test : total_test + batchsize, 1:] = pred
        total_test += batchsize
X = X.numpy()
np.save("DataCal", X)
