import torch
import torch.nn as nn
import torch.nn.functional as F

import road_signs_cnn.common as common
import road_signs_cnn.data as data


class ConvDownBlock(nn.Module):
    def __init__(self, *, inch, outch, kern, pad):
        super(ConvDownBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=inch,
            out_channels=outch,
            kernel_size=kern,
            padding=pad,

            # bias is not needed if is preceded by batch norm
            # (https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm)
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(num_features=outch)
        self.maxpool = nn.MaxPool2d(kernel_size=kern)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.maxpool(x)
        return x


# %%
model = nn.Sequential(
    # (*, 3, 32, 32) -> (*, 8, 16, 16)
    ConvDownBlock(inch=3, outch=8, kern=2, pad=1),
    # -> (*, 16, 8, 8)
    ConvDownBlock(inch=8, outch=16, kern=2, pad=1),
    # -> (*, 32, 4, 4)
    ConvDownBlock(inch=16, outch=32, kern=2, pad=1),
    #
    nn.Flatten(),
    nn.Dropout(0.2),
    nn.Linear(in_features=32 * 4 * 4, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=len(data.sign_names)),
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=common.LEARNING_RATE)