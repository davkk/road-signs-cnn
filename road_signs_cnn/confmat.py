import argparse
import re
from pathlib import Path

import numpy as np
import sklearn.metrics
import torch
from matplotlib import pyplot as plt

import road_signs_cnn.common as common
import road_signs_cnn.data as data
from road_signs_cnn.model import cnn

parser = argparse.ArgumentParser()
parser.add_argument(
    "checkpoint",
    metavar="string",
    type=Path,
    help="checkpoint path",
)
args = parser.parse_args()

checkpoint = str(args.checkpoint)

match_dropout = re.search("dropout=([0-9]+[.][0-9]+)", checkpoint)
dropout = (
    float(match_dropout.group(1))
    if match_dropout is not None
    else common.DROPOUT
)

model = cnn(dropout=dropout)

model.load_state_dict(torch.load(checkpoint))
model.eval()


def conf_mat(dataloader):
    n = len(data.sign_names)
    cm = np.zeros((n, n))
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs).argmax(dim=1)
            cm += sklearn.metrics.confusion_matrix(
                labels,
                outputs,
                labels=list(range(n)),
            )
    return cm


def main():
    cm = conf_mat(data.test_dl)
    plt.figure(figsize=(18, 10))

    plt.imshow(cm, cmap="gray_r")

    labels = list(range(len(data.sign_names)))
    plt.yticks(
        labels,
        labels=[f"{idx}: {label}" for idx, label in enumerate(data.sign_names)],
    )
    plt.xticks(labels)

    plt.tight_layout()
    plt.savefig(Path("figures") / "confmat.png", dpi=200)
    # plt.show()


if __name__ == "__main__":
    main()
