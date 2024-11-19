import torch

from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter


CONFIG_DIR = Path(__file__).parent
MODEL_DIR = CONFIG_DIR.parent
PROJ_DIR = MODEL_DIR.parent
RUNS_DIR = CONFIG_DIR / "runs"
DATASETS_DIR = PROJ_DIR / "data" / "datasets"

BATCH_SIZE = 128
LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-2
N_EPOCHS = 64
INPUT_DIM = 16064
LATENT_DIM = 2
HIDDEN_DIM = 8064
NUM_WORKERS = 0  # os.cpu_count() or 0
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
PIN_MEMORY = DEVICE == "cuda"
DATETIME_NOW = datetime.now().strftime("%Y%m%d-%H%M%S")
WRITER = SummaryWriter(RUNS_DIR / f"log_{DATETIME_NOW}")

print(
    f"SINGLE CONFIGURATION\nDevice: {DEVICE}\nBatch size: {BATCH_SIZE}\nConfiguration directory: {CONFIG_DIR.relative_to(PROJ_DIR)}\nRuns directory: {RUNS_DIR.relative_to(PROJ_DIR)}\nData directory: {DATASETS_DIR.relative_to(PROJ_DIR)}\n"
)
