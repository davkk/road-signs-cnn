import random
from pathlib import Path

import matplotlib.pyplot as plt

import road_signs_cnn.common as common
import road_signs_cnn.data as data

colors, markers = common.setup_pyplot()

fig, axes = plt.subplots(ncols=3, nrows=3)
axes = axes.reshape(-1)

random_images = list(zip(data.test_images.copy(), data.test_labels.copy()))
random.shuffle(random_images)
random_images = random_images[: len(axes)]

for (image, label), ax in zip(random_images, axes):
    ax.set_axis_off()
    ax.imshow(image)
    ax.set_title(data.sign_names[label])

fig.tight_layout()

plt.savefig(Path("figures") / "data_sample.png")

# plt.show()
