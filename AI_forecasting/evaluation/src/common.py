# common_eval_config.py
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = "AI_forecasting/data/input/masterdata_for_modeling.parquet"
OUT_BASE = Path("AI_forecasting/evaluation/advanced")
OUT_BASE.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "electricity_demand",
    "water_demand",
    "pm25",
    "congestion_index",
]

HORIZONS = [6, 12, 18, 24, 30, 36]
MAX_H = max(HORIZONS)

LOOKBACK = 36
DECODER_LEN = 12
EPS = 1e-6
SEED = 42
