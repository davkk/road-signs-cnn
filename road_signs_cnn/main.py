# %%
import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# %%
torch.manual_seed(2001)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# %%
class RoadSignsDataset(Dataset):
    def __init__(self, filename: str, root=Path("data")):
        self.sign_names: npt.NDArray[np.string_]
        _, self.sign_names = np.genfromtxt(
            root / "signname.csv", delimiter=",", skip_header=True, dtype="str"
        ).T

        metadata = pickle.load(open(root / filename, "rb"))
        self.images: npt.NDArray[np.int64] = metadata["features"]
        self.labels: npt.NDArray[np.string_] = metadata["labels"]

    def __len__(self):
        return len(self.images)

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


def main():
    test_ds = DataLoader(
        RoadSignsDataset("test.p"), batch_size=32, shuffle=False
    )
    train_ds = DataLoader(
        RoadSignsDataset("train.p"), batch_size=2000, shuffle=True
    )


if __name__ == "__main__":
    main()
