# %%
from utils_PlantNet300K import load_model
from torchvision.models import resnet50
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# %%
# load labels and id of the obs of each image
# these files are provided by Newplit.py
y = np.load("newlabels.npy")
obs_Id = np.load("Images_to_Obs.npy")

train_ind = np.load("train_ind.npy")


# Dataset is separated in two parts, the training data are splitted between them
# The class DoubleDataset manages this situation
class DoubleDataset(Dataset):
    def __init__(
        self, image_folder1, image_folder2, indices, labels, obs_Id=None, transform=None
    ):
        self.image_folder1 = image_folder1
        self.image_folder2 = image_folder2
        self.indices = indices
        self.labels = labels
        self.transform = transform
        self.obs_Id = obs_Id

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
        label = int(self.labels[real_idx])
        if self.transform:
            image = self.transform(image)

        if self.obs_Id:
            obs = self.obs_Id[real_idx]
            return image, label, obs
        else:
            return image, label


# %% Define transformation
image_size = 256
crop_size = 224
batch_size = 32
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

# Create DataSet and data loader
image_folder1 = "/PlantCLEF2015TestDataWithAnnotations/"
image_folder2 = "/PlantCLEF2015TrainingData/"

dataset = DoubleDataset(
    image_folder1, image_folder2, train_ind, y, transform=transform_train
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# %%
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %% Load model
filename = "resnet50_weights_best_acc.tar"  # pre-trained model path
use_gpu = False  # load weights on the gpu
model = resnet50(num_classes=1081)  # 1081 classes in Pl@ntNet-300K

load_model(model, filename=filename, use_gpu=use_gpu)


# %% Freeze all the layers except the last one
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1000)

# %% Define optimzer and criterion
criterion = nn.CrossEntropyLoss()  # nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# %% Train model
num_epochs = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%Metrics for validation curve
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_train += predicted.eq(labels).sum().item()
        total_train += labels.size(0)

    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(100 * correct_train / total_train)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accuracies[-1]:.2f}% | "
    )

# Save the model
torch.save(model.state_dict(), "model.pth")
