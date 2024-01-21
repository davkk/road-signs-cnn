from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import road_signs_cnn.common as common

colors, markers = common.setup_pyplot()

ROOT = Path("output")
lrs, epochs, trains, tests, _ = np.loadtxt(ROOT / "accuracy_vs_epoch.csv").T

data = defaultdict(list)

for lr, epoch, train, test in zip(lrs, epochs, trains, tests):
    data[lr].append((epoch, train, test))

plt.figure(figsize=(12, 8))

for idx, (lr, values) in enumerate(data.items()):
    epoch, train, test = zip(*values)
    trainplot = plt.plot(
        epoch,
        [train * 100 for train in train],
        "--",
        label=f"{lr:.0E} train",
        lw=2,
    )
    plt.plot(
        epoch,
        [test * 100 for test in test],
        "-",
        label=f"{lr:.0E} test",
        c=f"{trainplot[0].get_color()}",
        lw=2,
    )

plt.xlabel("Epoch")
plt.ylabel("Accuracy %")

plt.legend(bbox_to_anchor=(0.5, 1.2), loc="upper center", ncol=len(data))

plt.tight_layout()

plt.savefig(Path("figures") / "accuracy_lr_epoch.png")
# plt.show()
