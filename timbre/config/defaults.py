# Based off of https://github.com/maxrmorrison/deep-learning-project-template/blob/main/NAME/config/defaults.py

from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter

from .utils import get_device, should_pin_memory


# Configuration name
CONFIG = "defaults"


###############################################################################
# Data
###############################################################################

# Dataset names
DATASETS = ["nsynth"]

# EVALUATION_DATASETS = DATASETS


###############################################################################
# Directories
###############################################################################

_PACKAGE_DIR = Path(__file__).parent.parent

# For assets to bundle with pip release
# ASSETS_DIR = _PACKAGE_DIR / "assets"

_ROOT_DIR = _PACKAGE_DIR.parent

_DATA_DIR = _ROOT_DIR / "datasets"

# For preprocessed features
# CACHE_DIR = _DATA_DIR / "cache"

# For unprocessed datasets
SOURCES_DIR = _DATA_DIR / "sources"

# For preprocessed datasets
PREPROCESSED_DIR = _DATA_DIR / "preprocessed"

# For training and adaptation artifacts
RUNS_DIR = _ROOT_DIR / "runs"

# For evaluation artifacts
# EVAL_DIR = _ROOT_DIR.parent / "eval"


###############################################################################
# Training
###############################################################################


# Batch size per gpu
BATCH_SIZE = 128

# Steps between saving checkpoints
# CHECKPOINT_INTERVAL = 25000

# Training steps
# STEPS = 300000

# Worker threads for data loading
NUM_WORKERS = 8

# RANDOM_SEED = 1234

# [TODO] Formalize these options

LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 64
# INPUT_DIM = 16064
INPUT_DIM = 2048
LATENT_DIM = 2
# HIDDEN_DIM = 8064
HIDDEN_DIM = 1024

DEVICE = get_device()

USE_PIN_MEMORY = should_pin_memory()
DATETIME_NOW = datetime.now().strftime("%Y%m%d-%H%M%S")
WRITER = SummaryWriter(RUNS_DIR / f"log_{DATETIME_NOW}")


###############################################################################
# Evaluation
###############################################################################


# Steps between tensorboard logging
# EVALUATION_INTERVAL = 2500  # steps

# Steps to perform for tensorboard logging
# DEFAULT_EVALUATION_STEPS = 16
