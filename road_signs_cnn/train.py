import argparse
import sys

import numpy as np
import torch

import road_signs_cnn.common as common
import road_signs_cnn.data as data
from road_signs_cnn.model import cnn

parser = argparse.ArgumentParser()
parser.add_argument(
    "--lr",
    metavar="float",
    type=float,
    required=True,
    help="learning rate",
)
parser.add_argument(
    "--dropout",
    metavar="float",
    type=float,
    required=True,
    help="dropout",
)
args = parser.parse_args()


def main():
    lr: float = args.lr or common.LEARNING_RATE
    dropout: float = args.dropout or common.LEARNING_RATE

    model = cnn(dropout=dropout)

    loss_history = np.zeros(common.EPOCHS)
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, common.EPOCHS + 1):
        total_loss = 0

        for inputs, labels in data.train_dl:
            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history[epoch - 1] = total_loss

        print(epoch, total_loss)
        if epoch % 3 == 0:
            torch.save(
                model.state_dict(),
                common.CHECKPOINTS_ROOT
                / f"checkpoint_{lr=:.0E}_{dropout=:.1f}_{epoch=:02}.pt",
            )

    np.savetxt(
        common.OUTPUT_ROOT / f"loss_{lr=:.0E}_{dropout=:.2f}.csv",
        loss_history,
    )


if __name__ == "__main__":
    main()
