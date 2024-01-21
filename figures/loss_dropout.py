import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import road_signs_cnn.common as common

colors, markers = common.setup_pyplot()

ROOT = common.OUTPUT_ROOT / "loss_dropout"
files = os.listdir(ROOT)

dropouts = [
    float(re.search("dropout=([0-9]+[.][0-9]+)", file).group(1))
    for file in files
]

for idx, (file, dropout) in list(enumerate(zip(files, dropouts)))[::2]:
    data = np.loadtxt(ROOT / file)
    plt.plot(
        data,
        "-" + markers[idx % len(markers)],
        markersize=12,
        mew=3,
        label=f"{dropout:.1f}",
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(bbox_to_anchor=(0.5, 1.2), loc="upper center", ncol=len(dropouts))
plt.tight_layout()

plt.savefig(Path("figures") / "loss_dropout.png")
plt.show()
