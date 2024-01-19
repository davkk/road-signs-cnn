import matplotlib.pyplot as plt
import numpy as np

import road_signs_cnn.data as data

label, count = np.unique(data.train_labels, return_counts=True)
_, stemlines, baseline = plt.stem(label, count)

stemlines.set_linewidth(8)
baseline.set_color("none")

plt.xticks(list(range(len(data.sign_names))))
plt.tight_layout()
plt.show()
