import sys
from pathlib import Path

import torch

import road_signs_cnn.data as data
from road_signs_cnn.model import model

checkpoint = sys.argv[1]

model.load_state_dict(torch.load(checkpoint))
model.eval()


def accuracy(dataloader):
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs).argmax(dim=1)
            correct += (outputs == labels).sum().item()
    return correct / len(dataloader.dataset)


def main():
    print(
        Path(checkpoint).stem,
        accuracy(data.train_dl),
        accuracy(data.test_dl),
        accuracy(data.valid_dl),
    )


if __name__ == "__main__":
    main()
