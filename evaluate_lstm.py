#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# ──────────────────────────────────────────────────────────────────────────────
# EXACTLY the same constants you used in train_offline.py
# ──────────────────────────────────────────────────────────────────────────────
RAW_CSV         = 'raw_v2.csv'
WINDOW_SIZE     = 5                # must match train_offline.py
HORIZON_MINUTES = 30               # must match train_offline.py
FREQ_SECONDS    = 1                # sampling frequency used before
HORIZON_STEPS   = HORIZON_MINUTES * 60 // FREQ_SECONDS
FEATURE_COLS = [
    'pm05','pm1','pm025','pm04','pm10',
    'temperature','humidity','pressure','co2','particle_size',
    'occupancy','wall_particles','floor_particles','person_particles'
]
TARGET_COLS = ['pm05','pm1','pm025','pm04','pm10']
# ──────────────────────────────────────────────────────────────────────────────

def build_windows(df):
    """
    Given a sorted DataFrame `df`, build X (WINDOW_SIZE‐step windows of features)
    and Y (raw-targets at horizon) exactly as in train_offline.py.
    Returns:
      X: numpy array, shape = (n_samples, WINDOW_SIZE, n_features)
      Y: numpy array, shape = (n_samples, len(TARGET_COLS))
    """
    n = len(df)
    max_index = n - HORIZON_STEPS - WINDOW_SIZE
    if max_index <= 0:
        raise ValueError("Not enough data to build windows with given horizon/window.")
    X_list, y_list = [], []
    for i in range(max_index):
        # 1) feature window from i .. i+WINDOW_SIZE-1
        X_win = df.loc[i:i+WINDOW_SIZE-1, FEATURE_COLS].values
        # 2) index of target is (i + WINDOW_SIZE - 1 + HORIZON_STEPS)
        y_idx = i + WINDOW_SIZE - 1 + HORIZON_STEPS
        y_vals = df.loc[y_idx, TARGET_COLS].values
        X_list.append(X_win)
        y_list.append(y_vals)
    X = np.stack(X_list)   # shape = (n_samples, WINDOW_SIZE, n_features)
    Y = np.stack(y_list)   # shape = (n_samples, 5)
    return X, Y

def main():
    # 1) load raw CSV, sort by timestamp
    print("Loading raw data from", RAW_CSV, "…")
    df = pd.read_csv(RAW_CSV)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 2) build windows
    print("Building windows (horizon =", HORIZON_MINUTES, "minutes)…")
    X, Y_true = build_windows(df)
    n_samples = X.shape[0]
    print(f" → total samples: {n_samples}  (X shape = {X.shape}, Y shape = {Y_true.shape})")

    # 3) scale features the same way as during training
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    # flatten X so we can apply the same MinMax scaling
    ns, w, f = X.shape
    X_flat = X.reshape(ns * w, f)
    X_scaled = scaler_X.transform(X_flat).reshape(ns, w, f)
    Y_scaled = scaler_y.transform(Y_true)

    # 4) split 80/20 exactly as train_offline did
    split_idx = int(0.8 * ns)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    Y_train, Y_val = Y_scaled[:split_idx], Y_scaled[split_idx:]
    Y_true_val     = Y_true[split_idx:]  # keep the *unscaled* ground truth for evaluation

    print(f"→ train samples: {X_train.shape[0]},  val samples: {X_val.shape[0]}")

    # 5) load the trained LSTM
    print("Loading model from best_lstm.keras …")
    model = load_model("best_lstm.keras")

    # 6) predict on validation set
    print("Running predictions on validation set…")
    Y_pred_scaled = model.predict(X_val, verbose=0)  # shape = (val_size, 5)
    # invert‐scale
    Y_pred = scaler_y.inverse_transform(Y_pred_scaled)

    # 7) compute and print RMSE/MAE per target
    print()
    for i, col in enumerate(TARGET_COLS):
        mse = mean_squared_error(Y_true_val[:, i], Y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_true_val[:, i], Y_pred[:, i])
        print(f"Target = {col:6s}  →  RMSE = {rmse:8.3f},   MAE = {mae:8.3f}")

    # 8) also print an overall summary
    overall_rmse = np.sqrt(mean_squared_error(Y_true_val.reshape(-1), Y_pred.reshape(-1)))
    overall_mae  = mean_absolute_error(Y_true_val.reshape(-1), Y_pred.reshape(-1))
    print(f"\nOverall (all 5 targets combined):  RMSE = {overall_rmse:.3f},  MAE = {overall_mae:.3f}")

if __name__ == "__main__":
    main()
