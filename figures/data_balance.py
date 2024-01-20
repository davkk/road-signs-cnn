from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import road_signs_cnn.common as common
import road_signs_cnn.data as data

colors, markers = common.setup_pyplot()

plt.figure(figsize=(12, 12))

label, count = np.unique(data.train_labels, return_counts=True)
_, stemlines, baseline = plt.stem(label, count, orientation="horizontal")

stemlines.set_linewidth(8)
baseline.set_color("none")

plt.yticks(
    label,
    labels=[f"{idx+1}: {name}" for idx, name in enumerate(data.sign_names)],
)
plt.tight_layout()

plt.savefig(Path("figures") / "data_balance.png", dpi=100)

# plt.show()
