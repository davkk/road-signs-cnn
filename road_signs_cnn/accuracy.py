import argparse
import re
from pathlib import Path

import torch

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
