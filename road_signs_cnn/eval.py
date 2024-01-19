import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import road_signs_cnn.common as common
import road_signs_cnn.data as data
from road_signs_cnn.model import model

model.load_state_dict(torch.load(common.CHECKPOINTS_ROOT / "checkpoint.pt"))
model.eval().to(common.device)


def accuracy(dataloader):
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs).argmax(dim=1)
            correct += (outputs == labels).sum().item()
    return correct / len(dataloader.dataset)


def conf_mat(dataloader, classes):
    cm = np.zeros((classes, classes))
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(common.device)
            outputs = model(inputs).argmax(dim=1).cpu()
            cm += confusion_matrix(labels, outputs, labels=list(range(classes)))
    return cm


if __name__ == "__main__":
    print(f"{accuracy(data.train_dl)=}, {accuracy(data.test_dl)=}")

    cm = conf_mat(data.test_dl, len(data.sign_names))

    fig, ax = plt.subplots(1, 1)
    ax.imshow(cm, cmap="gray_r")

    ax.set_xticks(list(range(len(data.sign_names))))
    ax.set_yticks(list(range(len(data.sign_names))))

    plt.tight_layout()
    plt.show()
