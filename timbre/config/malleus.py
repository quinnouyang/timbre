from pathlib import Path

MODULE = "timbre"

# Configuration name
CONFIG = "malleus"

_DATA_DIR = Path("/") / "mnt" / "data" / "quinn" / "timbre"

# For unprocessed datasets
SOURCES_DIR = _DATA_DIR / "sources"

# For preprocessed features
PREPROCESSED_DIR = _DATA_DIR / "preprocessed"
