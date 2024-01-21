from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import road_signs_cnn.common as common

colors, markers = common.setup_pyplot()

ROOT = Path("output")
data = np.loadtxt(ROOT / "accuracy_dropout.csv")
_, dropouts, epochs, trains, tests, _ = data[data[:, 2] == 15].T

trainplot = plt.plot(
    dropouts,
    trains * 100,
    "--",
    lw=2,
    label="train accuracy",
)
plt.plot(
    dropouts,
    tests * 100,
    "-",
    lw=2,
    c=f"{trainplot[0].get_color()}",
    label="test accuracy",
)
plt.fill_between(dropouts, tests * 100, trains * 100, color="pink", alpha=0.3)

plt.xlabel("Dropout")
plt.ylabel("Accuracy %")

plt.legend()
plt.tight_layout()

plt.savefig(Path("figures") / "accuracy_dropout.png")
# plt.show()
