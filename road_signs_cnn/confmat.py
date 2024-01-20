import sys

import numpy as np
import sklearn.metrics
import torch
from matplotlib import pyplot as plt

import road_signs_cnn.data as data
from road_signs_cnn.model import model

checkpoint = sys.argv[1]

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

    plt.imshow(cm, cmap="gray_r")

    labels = list(range(len(data.sign_names)))
    plt.yticks(
        labels,
        labels=[f"{idx}: {label}" for idx, label in enumerate(data.sign_names)],
    )
    plt.xticks(labels)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
