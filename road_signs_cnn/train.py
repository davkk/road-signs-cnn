import numpy as np
import torch

import road_signs_cnn.common as common
import road_signs_cnn.data as data
from road_signs_cnn.model import loss_fn, model, optimizer

loss_history = np.zeros(common.EPOCHS)

model.train()

for epoch in range(common.EPOCHS):
    total_loss = 0

    for inputs, labels in data.train_dl:
        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    loss_history[epoch] = total_loss

    print(epoch + 1, total_loss)
    if epoch % 3:
        torch.save(
            model.state_dict(),
            common.CHECKPOINTS_ROOT / f"checkpoint_epoch={epoch + 1}.pt",
        )

np.savetxt(
    common.OUTPUT_ROOT / f"loss_lr={common.LEARNING_RATE}.csv",
    loss_history,
)
