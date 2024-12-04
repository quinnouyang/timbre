# Based off of https://github.com/maxrmorrison/deep-learning-project-template/blob/main/NAME/config/defaults.py

import torch

from os import cpu_count
from pathlib import Path

# Configuration name
CONFIG = "timbre"


###############################################################################
# Data
###############################################################################

# Dataset names
DATASETS = ["nsynth"]

EVALUATION_DATASETS = DATASETS


###############################################################################
# Directories
###############################################################################

PACKAGE_DIR = Path(__file__).parent.parent

# For assets to bundle with pip release
# ASSETS_DIR = PACKAGE_DIR / "assets"

ROOT_DIR = PACKAGE_DIR.parent

# For preprocessed features
# CACHE_DIR = ROOT_DIR.parent / "data" / "cache"

# For unprocessed datasets
SOURCES_DIR = ROOT_DIR / "data" / "sources"

# For preprocessed datasets
DATA_DIR = ROOT_DIR / "data" / "datasets"

# For training and adaptation artifacts
RUNS_DIR = ROOT_DIR / "runs"

# For evaluation artifacts
# EVAL_DIR = ROOT_DIR.parent / "eval"


###############################################################################
# Training
###############################################################################


# Batch size per gpu
BATCH_SIZE = 128

# Steps between saving checkpoints
CHECKPOINT_INTERVAL = 25000

# Training steps
STEPS = 300000

n_cpus = cpu_count() or 0
if not n_cpus:
    raise ValueError("Could not determine the number of CPUs")

# Worker threads for data loading
NUM_WORKERS = int(n_cpus / max(1, torch.cuda.device_count()))

RANDOM_SEED = 1234

# [TODO] Formalize these options

LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-2
N_EPOCHS = 64
INPUT_DIM = 16064
LATENT_DIM = 2
HIDDEN_DIM = 8064

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

USE_PIN_MEMORY = DEVICE == "cuda"
# DATETIME_NOW = datetime.now().strftime("%Y%m%d-%H%M%S")
# WRITER = SummaryWriter(RUNS_DIR / f"log_{DATETIME_NOW}")


###############################################################################
# Evaluation
###############################################################################


# Steps between tensorboard logging
# EVALUATION_INTERVAL = 2500  # steps

# Steps to perform for tensorboard logging
# DEFAULT_EVALUATION_STEPS = 16
