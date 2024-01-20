# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("output")
lrs = os.listdir(ROOT)

for lr in lrs:
    data = np.loadtxt(ROOT / lr)
    plt.plot(data, label=lr.split(".")[0].split("=")[1])

plt.legend()
plt.tight_layout()
plt.show()
