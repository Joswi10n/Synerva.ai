# Zone.py
import pandas as pd, numpy as np
from collections import deque
from sklearn.preprocessing import MinMaxScaler
# ... your cyc(), add_roll() helpers, TARGETS, BASE_FEATURES, HORIZON_SEC, etc.

class Zone:
    def __init__(self, z, model, sx:MinMaxScaler, sy:MinMaxScaler, feats, write_fn):
        self.z        = z
        self.m        = model
        self.sx, self.sy = sx, sy
        self.feats    = feats
        self.buf      = deque(maxlen=WINDOW_BINS)
        self.pending  = deque()
        self.train_X, self.train_y = [], []
        self.val_X, self.val_y     = deque(maxlen=VAL_HOLD), deque(maxlen=VAL_HOLD)
        self.last_min = None
        self.write_fn = write_fn  # injection of your Influx or Kafka writer

    def push_record(self, rec):
        """Push a new raw record (dict) with 'timestamp' in ms, a 'zone' key, and all BASE_FEATURES + TARGETS."""
        ts = pd.to_datetime(rec["timestamp"], unit="ms").floor("min")
        self.buf.append((ts, rec))
        # queue ground truth once window is full
        if len(self.buf) == WINDOW_BINS:
            y_true = np.array([rec[t] for t in TARGETS])[None, :]
            y_norm = self.sy.transform(y_true)[0]
            wnd0  = int(self.buf[0][0].timestamp())
            self.pending.append((wnd0, self.build_X(), y_norm))
        # on minute‐tick, predict
        if ts != self.last_min:
            self.last_min = ts
            self.predict_and_write(ts)

    def build_X(self):
        rows = []
        for ts, rec in self.buf:
            d = {c: rec[c] for c in BASE_FEATURES}
            d.update(dict(zip(["hour_sin","hour_cos","dow_sin","dow_cos"], cyc(ts))))
            rows.append(d)
        df = pd.DataFrame(rows)
        df = add_roll(df, TARGETS, WINDOW_BINS)
        X_raw = df[self.feats].values.astype(np.float32)
        return self.sx.transform(X_raw)

    def predict_and_write(self, ts_minute):
        if len(self.buf) < WINDOW_BINS: return
        X = self.build_X()[None, ...]
        y_norm = self.m.predict(X, verbose=0)[0]
        y_pred = self.sy.inverse_transform(y_norm[None, :])[0]
        future_ts = ts_minute + pd.Timedelta(seconds=HORIZON_SEC)
        # call your writer
        self.write_fn(zone=self.z, timestamp=future_ts.to_pydatetime(), 
                      values=dict(zip(TARGETS, map(float, y_pred))))
        print(f"[{self.z}] Pred @ {future_ts} → {y_pred}")

    def try_collect_gt(self, now_sec):
        while self.pending and (self.pending[0][0] + HORIZON_SEC) <= now_sec:
            _, Xn, yn = self.pending.popleft()
            self.train_X.append(Xn); self.train_y.append(yn)
            self.val_X.append(Xn);    self.val_y.append(yn)
            if len(self.train_X) >= TRAIN_BATCH:
                self._train_step()

    def _train_step(self):
        Xb = np.stack(self.train_X); yb = np.stack(self.train_y)
        self.train_X.clear(); self.train_y.clear()
        hist = self.m.fit(Xb, yb, epochs=1, batch_size=TRAIN_BATCH, verbose=0)
        print(f"[{self.z}] Online train loss={hist.history['loss'][0]:.3f}")
        if len(self.val_X) == VAL_HOLD:
            Xv, yv = np.stack(self.val_X), np.stack(self.val_y)
            l,m = self.m.evaluate(Xv, yv, verbose=0)
            print(f"[{self.z}] Val({VAL_HOLD}) loss={l:.3f} mae={m:.3f}")
            self.val_X.clear(); self.val_y.clear()
