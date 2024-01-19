import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import road_signs_cnn.common as common
import road_signs_cnn.data as data

sign_names: npt.NDArray[np.string_]
_, sign_names = np.genfromtxt(
    common.DATA_ROOT / "signname.csv",
    delimiter=",",
    skip_header=True,
    dtype="str",
).T


def load_data(filename: str):
    metadata = pickle.load(open(common.DATA_ROOT / filename, "rb"))
    images: npt.NDArray[np.int64] = metadata["features"]
    labels: npt.NDArray[np.int64] = metadata["labels"]
    return images, labels


class RoadSignsDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        transform = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(3, scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(
                    # ImageNet's commonly used mean and std values
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        image = transform(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        return image, label


train_images, train_labels = load_data("train.p")
test_images, test_labels = load_data("test.p")
valid_images, valid_labels = load_data("valid.p")

train_dl = DataLoader(
    RoadSignsDataset(images=train_images, labels=train_labels),
    batch_size=32,
    shuffle=True,
    num_workers=5,
)
test_dl = DataLoader(
    RoadSignsDataset(images=test_images, labels=test_labels),
    batch_size=2000,
    shuffle=False,
    num_workers=5,
)
valid_dl = DataLoader(
    RoadSignsDataset(images=valid_images, labels=valid_labels),
    batch_size=2000,
    shuffle=False,
    num_workers=5,
)


def show_image_sample():
    fig, axes = plt.subplots(ncols=3, nrows=3)
    axes = axes.reshape(-1)

    random_images = list(zip(data.test_images.copy(), data.test_labels.copy()))
    random.shuffle(random_images)
    random_images = random_images[: len(axes)]

    for (image, label), ax in zip(random_images, axes):
        ax.set_axis_off()
        ax.imshow(image)
        ax.set_title(data.sign_names[label])

    fig.tight_layout()
    plt.show()


def show_data_balance():
    label, count = np.unique(train_labels, return_counts=True)
    _, stemlines, baseline = plt.stem(label, count)

    stemlines.set_linewidth(8)
    baseline.set_color("none")

    plt.xticks(list(range(len(sign_names))))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_image_sample()
    show_data_balance()
