# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import road_signs_cnn.common as common

colors, markers = common.setup_pyplot()

ROOT = common.OUTPUT_ROOT / "loss_lr"
lrs = os.listdir(ROOT)

for idx, lr in enumerate(sorted(lrs)):
    data = np.loadtxt(ROOT / lr)
    plt.plot(
        data,
        "-" + markers[idx % len(markers)],
        markersize=12,
        mew=3,
        label=lr.split(".")[0].split("=")[1],
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(bbox_to_anchor=(0.5, 1.2), loc="upper center", ncol=len(lrs))
plt.tight_layout()

plt.savefig(Path("figures") / "loss_lr.png")
# plt.show()
