from pathlib import Path

import matplotlib.pyplot as plt
import torch

OUTPUT_ROOT = Path("output")
DATA_ROOT = Path("data")
CHECKPOINTS_ROOT = Path("checkpoints")

EPOCHS = 15
LEARNING_RATE = 1e-3
DROPOUT = 0.2


torch.manual_seed(2001)
device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_pyplot():
    plt.style.use("tableau-colorblind10")

    plt.rcParams["figure.figsize"] = (9, 6)
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    plt.rcParams["axes.formatter.limits"] = -3, 3
    plt.rcParams["axes.grid"] = False

    plt.rc("font", size=14)  # controls default text sizes
    plt.rc("axes", titlesize=14)  # fontsize of the axes title
    plt.rc("axes", labelsize=16)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=14)  # fontsize of the tick labels
    plt.rc("legend", fontsize=14)  # legend fontsize
    plt.rc("figure", titlesize=22)  # fontsize of the figure title

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["+", "1", "o", "2", "*", "3"]

    return colors, markers
