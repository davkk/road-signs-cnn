# %% [markdown]
# # Road Signs CNN

# %% [markdown]
# ## Setup

# %%
import pickle
import random
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# %%
torch.manual_seed(2001)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] {device=}")

DATA_ROOT = Path("data")
LEARNING_RATE = 1e-4

# %% [markdown]
# ## Load the data

# %%
sign_names: npt.NDArray[np.string_]
_, sign_names = np.genfromtxt(
    DATA_ROOT / "signname.csv", delimiter=",", skip_header=True, dtype="str"
).T


def load_data(filename: str):
    metadata = pickle.load(open(DATA_ROOT / filename, "rb"))
    images: npt.NDArray[np.int64] = metadata["features"]
    labels: npt.NDArray[np.int64] = metadata["labels"]
    return images, labels


# %%
train_images, train_labels = load_data("train.p")
test_images, test_labels = load_data("test.p")
valid_images, valid_labels = load_data("valid.p")

# %%
fig, axes = plt.subplots(ncols=3, nrows=3)
axes = axes.reshape(-1)

random_images = list(zip(test_images.copy(), test_labels.copy()))
random.shuffle(random_images)
random_images = random_images[: len(axes)]

for (image, label), ax in zip(random_images, axes):
    ax.imshow(image)
    ax.set_title(sign_names[label])

fig.tight_layout()
# plt.show()


# %% [markdown]
# ## Create custom data loaders


# %%
class RoadSignsDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    # ImageNet commonly used mean and std values
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        image = transform(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        return image, label


# %%
class ConvDownBlock(nn.Module):
    def __init__(self, *, inch, outch, kern, pad):
        super(ConvDownBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=inch,
            out_channels=outch,
            kernel_size=kern,
            padding=pad,
        )
        self.bn = nn.BatchNorm2d(num_features=outch)
        self.maxpool = nn.MaxPool2d(kernel_size=kern)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.maxpool(x)
        return x


# %%
train_dl = DataLoader(
    RoadSignsDataset(images=train_images, labels=train_labels),
    batch_size=32,
    shuffle=True,
)
test_dl = DataLoader(
    RoadSignsDataset(images=test_images, labels=test_labels),
    batch_size=2000,
    shuffle=False,
)
valid_dl = DataLoader(
    RoadSignsDataset(images=valid_images, labels=valid_labels),
    batch_size=2000,
    shuffle=False,
)

# %%
db = ConvDownBlock(inch=3, outch=8, kern=2, pad=1)
x = torch.randn(10, 3, 32, 32)
x.shape, db(x).shape


# %%
model = nn.Sequential(
    # (*, 3, 32, 32) -> (*, 8, 16, 16)
    ConvDownBlock(inch=3, outch=8, kern=2, pad=1),
    # -> (*, 16, 8, 8)
    ConvDownBlock(inch=8, outch=16, kern=2, pad=1),
    # -> (*, 32, 4, 4)
    ConvDownBlock(inch=16, outch=32, kern=2, pad=1),
    #
    nn.Flatten(),
    nn.Dropout(0.2),
    nn.Linear(in_features=32 * 4 * 4, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=len(train_dl)),
)

# %%
model(torch.randn(3, 3, 32, 32))

# %% [markdown]
# ## Train the model

# %%
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# %%
def train_loop():
    total_loss = 0

    for X, y in train_dl:
        opt.zero_grad()

        y_pred = model(X)

        loss = loss_fn(y_pred, y)

        loss.backward()
        opt.step()
        total_loss += loss.item()

    return total_loss


# %%
loss_history = []

# %%
epochs = 5
for epoch in range(epochs):
    loss = train_loop()
    loss_history.append(loss)
    if not epoch % 10:
        print(f"Epoch: {epoch} loss: {loss}")

# %%
plt.plot(loss_history)
plt.show()

# %% [markdown]
# ## Evaluate the model


# %%
def accuracy(dataloader):
    correct = 0
    all = len(dataloader.dataset)

    with torch.no_grad():
        for X, y in dataloader:
            y_pred = model(X).argmax(dim=1)
            correct += (y_pred == y).sum().item()
    return correct / all


accuracy(train_dl), accuracy(test_dl)


# %%
def my_confusion_matrix(dataloader, classes):
    cm = np.zeros((classes, classes))
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y_pred = model(X).argmax(dim=1).cpu()
            cm += confusion_matrix(y, y_pred, labels=list(range(classes)))
    return cm


# %%
cm = my_confusion_matrix(test_dl, len(sign_names))
fig, ax = plt.subplots(1, 1)
ax.imshow(cm, cmap="gray_r")

ax.set_xticks(list(range(len(sign_names))))
ax.set_xticklabels(sign_names, rotation=90)
ax.set_yticks(list(range(10)))
ax.set_yticklabels(sign_names)

plt.show()

# %% [markdown]
# ## Save the model

# %%
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(),
    },
    "my-checkpoint.pkl",
)
