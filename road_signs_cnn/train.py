import sys

import numpy as np
import torch

import road_signs_cnn.common as common
import road_signs_cnn.data as data
from road_signs_cnn.model import loss_fn, model, optimizer


def main():
    lr = sys.argv[1] if len(sys.argv) == 2 else common.LEARNING_RATE

    loss_history = np.zeros(common.EPOCHS)
    model.train()

    for epoch in range(1, common.EPOCHS + 1):
        total_loss = 0

        for inputs, labels in data.train_dl:
            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history[epoch] = total_loss

        print(epoch, total_loss)
        if epoch % 3 == 0:
            torch.save(
                model.state_dict(),
                common.CHECKPOINTS_ROOT
                / f"checkpoint_lr={lr:.0E}_epoch={epoch:02}.pt",
            )

    np.savetxt(
        common.OUTPUT_ROOT / f"loss_lr={lr:.0E}.csv",
        loss_history,
    )


if __name__ == "__main__":
    main()
