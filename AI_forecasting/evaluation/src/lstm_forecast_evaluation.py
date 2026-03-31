#!/usr/bin/env python3
"""
lstm_multivariate_seq2seq_rolling_eval.py

GLOBAL MULTIVARIATE SEQ2SEQ LSTM
ROLLING-ORIGIN EVALUATION
RESEARCH-GRADE IMPLEMENTATION
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
DATA_PATH = Path("AI_forecasting/data/input/masterdata_for_modeling_lstm.parquet")
OUT_DIR = Path("AI_forecasting/evaluation/lstm_multivariate_seq2seq")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "electricity_demand",
    "water_demand",
    "congestion_index",
    "pm25",
]

EXOG_FEATURES = [
    "rainfall",
    "population",
    "t_idx_norm",
    "month_sin",
    "month_cos",
    "is_monsoon",
]

TRAIN_ENDS = pd.to_datetime([
    "2019-12-01",
    "2020-12-01",
    "2021-12-01",
    "2022-12-01",
    "2023-12-01",
])

HORIZONS = [12, 36, 60]
MAX_H = max(HORIZONS)

LOOKBACK = 36
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
HIDDEN_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-6
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
# --------------------------------------


# ---------------- DATASET ----------------
class Seq2SeqDataset(Dataset):
    """
    X(t-lookback+1:t) -> y(t+1:t+H)
    Multivariate encoder, vector decoder
    """
    def __init__(self, X, y, lookback, horizon):
        self.X, self.y = [], []

        for i in range(len(X) - lookback - horizon + 1):
            self.X.append(X[i:i + lookback])
            self.y.append(y[i + lookback:i + lookback + horizon])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# ---------------- MODEL ----------------
class Seq2SeqLSTM(nn.Module):
    def __init__(self, n_features, hidden_size, horizon):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        return self.decoder(h[-1])


# ---------------- LOAD DATA ----------------
df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)

rows = []

# ---------------- EVALUATION ----------------
for target in TARGETS:
    print(f"\nEvaluating MULTIVARIATE SEQ2SEQ LSTM for target: {target}")

    FEATURES = [target] + EXOG_FEATURES
    n_features = len(FEATURES)

    for train_end in TRAIN_ENDS:
        print(f"  Train end: {train_end.date()}")

        datasets = []

        for ward, g in df.groupby("ward_id"):
            g = g.set_index("date")
            train = g.loc[:train_end]

            if len(train) < LOOKBACK + MAX_H + 1:
                continue

            data = train[FEATURES].astype(float)
            if data[target].var() < EPS:
                continue

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_scaled = scaler_X.fit_transform(data.values)
            y_scaled = scaler_y.fit_transform(
                data[[target]].values
            ).flatten()

            ds = Seq2SeqDataset(
                X_scaled,
                y_scaled,
                LOOKBACK,
                MAX_H
            )

            if len(ds) > 0:
                datasets.append((ds, scaler_X, scaler_y, ward, g))

        if not datasets:
            continue

        loader = DataLoader(
            torch.utils.data.ConcatDataset([d[0] for d in datasets]),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        model = Seq2SeqLSTM(
            n_features=n_features,
            hidden_size=HIDDEN_SIZE,
            horizon=MAX_H
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        # ---- TRAIN ----
        model.train()
        for _ in range(EPOCHS):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

        model.eval()

        # ---- TEST ----
        for _, scaler_X, scaler_y, ward, g in datasets:
            history = g.loc[:train_end]
            if len(history) < LOOKBACK:
                continue

            X_hist = history[FEATURES].astype(float).values
            X_scaled = scaler_X.transform(X_hist)

            x_last = X_scaled[-LOOKBACK:]
            x_tensor = torch.tensor(
                x_last.reshape(1, LOOKBACK, n_features),
                dtype=torch.float32
            ).to(DEVICE)

            with torch.no_grad():
                preds_scaled = model(x_tensor).cpu().numpy()[0]

            preds = scaler_y.inverse_transform(
                preds_scaled.reshape(-1, 1)
            ).flatten()

            for H in HORIZONS:
                test_date = train_end + pd.offsets.MonthBegin(H)
                if test_date not in g.index:
                    continue

                y_true = float(g.loc[test_date, target])
                y_pred = preds[H - 1]

                rows.append({
                    "model": "LSTM_SEQ2SEQ_MULTI",
                    "target": target,
                    "ward_id": ward,
                    "train_end": train_end,
                    "horizon": H,
                    "mae": abs(y_true - y_pred),
                    "rmse": (y_true - y_pred) ** 2,
                    "mape_pct": abs(
                        (y_true - y_pred) /
                        max(abs(y_true), EPS)
                    ) * 100,
                })


# ---------------- SAVE RESULTS ----------------
df_metrics = pd.DataFrame(rows)

df_metrics.to_csv(
    OUT_DIR / "lstm_multivariate_seq2seq_split_level_metrics.csv",
    index=False
)

summary = (
    df_metrics
    .groupby(["model", "target", "horizon"])
    .agg(
        mae=("mae", "mean"),
        rmse=("rmse", lambda x: np.sqrt(np.mean(x))),
        mape_pct=("mape_pct", "mean"),
        n_splits=("mae", "count"),
    )
    .reset_index()
)

summary.to_csv(
    OUT_DIR / "lstm_multivariate_seq2seq_horizon_averaged_metrics.csv",
    index=False
)

print("\n✅ MULTIVARIATE SEQ2SEQ LSTM evaluation complete")
