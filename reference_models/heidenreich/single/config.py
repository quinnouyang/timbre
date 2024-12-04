import torch

from datetime import datetime
from os import cpu_count
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Lambda


class FlattenNormalize(torch.nn.Module):
    def forward(self, x):
        return x.view(-1) - 0.5


TRANSFORM = Compose([ToImage(), ToDtype(torch.float32, scale=True), FlattenNormalize()])

CONFIG_DIR = Path(__file__).parent
MODEL_DIR = CONFIG_DIR.parent
PROJ_DIR = MODEL_DIR.parent.parent
RUNS_DIR = CONFIG_DIR / "runs"
DATASETS_DIR = MODEL_DIR / "datasets"

BATCH_SIZE = 2048
LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-2
N_EPOCHS = 64
INPUT_DIM = 784
LATENT_DIM = 2
HIDDEN_DIM = 512

NUM_WORKERS = cpu_count() or 0
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
PIN_MEMORY = DEVICE == "cuda"
DATETIME_NOW = datetime.now().strftime("%Y%m%d-%H%M%S")
WRITER = SummaryWriter(RUNS_DIR / f"log_{DATETIME_NOW}")

print(
    f"SINGLE v2 CONFIGURATION\nDevice: {DEVICE}\nBatch size: {BATCH_SIZE}\nConfiguration directory: {CONFIG_DIR.relative_to(PROJ_DIR)}\nRuns directory: {RUNS_DIR.relative_to(PROJ_DIR)}\nData directory: {DATASETS_DIR.relative_to(PROJ_DIR)}\n"
)
