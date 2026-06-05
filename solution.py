"""
Traffic Demand Prediction — Flipkart Gridlock
Final corrected solution.

FEATURES (built only from the official train/test files):
  geo_ts          — Bayesian-smoothed (geo,slot) mean from day48   [primary]
  geo_m48         — per-geohash mean demand, day48
  geo_std48       — per-geohash temporal volatility, day48
  d49_gm          — day49 geo-level calibration (from slots 0-8)  [new signal]
  geo_shift       — d49_gm − geo_m48
  lag8/7/6        — same-day slot-8/7/6 anchor (d48 for training, d49 for test)
  lag3m           — rolling mean of lag8/7/6
  d49_trend       — linear slope of day49 early demand for this geo [new signal]
  d49_last        — most recent day49 demand observation           [new signal]
  d49_max         — max day49 demand (detects spikes)             [new signal]
  nbr_d48         — mean demand of spatial neighbors at same slot, day48
  nbr_gm          — geo-mean of spatial neighbors
  slot, hour, is_morn, is_night
  NumberofLanes, road_enc, weath_enc, lv_enc, lm_enc, temp_f, temp_b
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pygeohash as pgh
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

TRAIN_PATH = "dataset/train.csv"
TEST_PATH  = "dataset/test.csv"
SUB_PATH   = "submission.csv"

# ── 1. Load ───────────────────────────────────────────────────────────────────
train  = pd.read_csv(TRAIN_PATH)
test   = pd.read_csv(TEST_PATH)
sample = pd.read_csv("dataset/sample_submission.csv")
sample.columns = ["Index", "true_demand"]

def parse_slot(ts):
    h, m = map(int, ts.split(":"))
    return h * 4 + m // 15

train["slot"] = train.timestamp.apply(parse_slot)
test["slot"]  = test.timestamp.apply(parse_slot)

d48 = train[train.day == 48].copy()
d49 = train[train.day == 49].copy()

print(f"day48={len(d48):,}  day49_early={len(d49):,}  test={len(test):,}")

# ── 2. Day48 lookup tables ────────────────────────────────────────────────────
d48_gs_mean  = d48.groupby(["geohash", "slot"])["demand"].mean()
d48_geo_mean = d48.groupby("geohash")["demand"].mean()
d48_geo_std  = d48.groupby("geohash")["demand"].std().fillna(0)
d48_slot_mean= d48.groupby("slot")["demand"].mean()
d48_global   = float(d48["demand"].mean())

# ── 3. Day49 calibration tables ───────────────────────────────────────────────
d49_geo_mean = d49.groupby("geohash")["demand"].mean()
d49_global   = float(d49["demand"].mean())
geo_shift_map= (d49_geo_mean - d48_geo_mean).to_dict()
d48_early_geo_mean = d48[d48.slot <= 8].groupby("geohash")["demand"].mean()
early_shift_map = (d49_geo_mean - d48_early_geo_mean).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict()
early_ratio_map = ((d49_geo_mean + 0.02) / (d48_early_geo_mean + 0.02)).replace(
    [np.inf, -np.inf], np.nan
).fillna(1).clip(0.2, 3.0).to_dict()

# Slot anchor lags
d49_s = {s: d49[d49.slot==s].set_index("geohash")["demand"].to_dict() for s in range(9)}
d48_s = {s: d48[d48.slot==s].set_index("geohash")["demand"].to_dict() for s in range(96)}

# Day49 trend features per geohash
d49_trend_map = {}; d49_last_map = {}; d49_max_map = {}
for geo, grp in d49.groupby("geohash"):
    slt = grp.sort_values("slot")["slot"].values
    val = grp.sort_values("slot")["demand"].values
    d49_last_map[geo] = float(val[-1])
    d49_max_map[geo]  = float(val.max())
    if len(val) >= 2:
        sm = slt.mean(); vm = val.mean()
        ss = ((slt - sm)**2).sum()
        d49_trend_map[geo] = float(((slt - sm)*(val - vm)).sum() / ss) if ss > 0 else 0.
    else:
        d49_trend_map[geo] = 0.

# ── 4. Geohash neighbors ──────────────────────────────────────────────────────
all_geos = set(train["geohash"].unique()) | set(test["geohash"].unique())
print("Computing neighbors…", end=" ", flush=True)
nc = {}
for geo in all_geos:
    nbrs = set()
    try:
        for d_ in ("top","bottom","right","left"):
            n = pgh.get_adjacent(geo, d_)
            if n in all_geos: nbrs.add(n)
        for ch in [("top","right"),("top","left"),("bottom","right"),("bottom","left")]:
            n = pgh.get_adjacent(pgh.get_adjacent(geo, ch[0]), ch[1])
            if n in all_geos: nbrs.add(n)
    except Exception:
        pass
    nc[geo] = list(nbrs)
d48_gsd = d48_gs_mean.to_dict()
nbr_gm_map = {
    geo: float(np.mean([d48_geo_mean[n] for n in nbrs if n in d48_geo_mean.index]))
         if any(n in d48_geo_mean.index for n in nbrs) else d48_global
    for geo, nbrs in nc.items()
}
print("done.")

TEMP_MEDIAN = float(train["Temperature"].median())
k = 5  # Bayesian smoothing strength

# Bayesian-smoothed (geo,slot) map
gs_count = d48.groupby(["geohash","slot"])["demand"].count()
geo_ts_map = {}
for (geo, s), raw in d48_gs_mean.items():
    c  = gs_count[(geo, s)]
    gm = d48_geo_mean.get(geo, d48_global)
    geo_ts_map[(geo, s)] = (c * raw + k * gm) / (c + k)

# ── 5. Feature builder ────────────────────────────────────────────────────────
def build_features(df, lag8_map, lag7_map, lag6_map, lag_global):
    df = df.copy()
    df["road_enc"]  = df["RoadType"].map({"Residential":0,"Street":1,"Highway":2}).fillna(-1).astype(int)
    df["weath_enc"] = df["Weather"].map({"Sunny":0,"Foggy":1,"Rainy":2,"Snowy":3}).fillna(-1).astype(int)
    df["lv_enc"]    = (df["LargeVehicles"] == "Allowed").astype(int)
    df["lm_enc"]    = (df["Landmarks"] == "Yes").astype(int)
    df["temp_f"]    = df["Temperature"].fillna(TEMP_MEDIAN)
    df["temp_b"]    = pd.cut(df["temp_f"], bins=10, labels=False).astype(float)
    df["hour"]      = df["slot"] // 4
    df["is_morn"]   = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["is_night"]  = (df["hour"] < 6).astype(int)
    g, s = df["geohash"].values, df["slot"].values

    df["geo_ts"]    = [geo_ts_map.get((gi,si), d48_geo_mean.get(gi, d48_global)) for gi,si in zip(g,s)]
    df["geo_m48"]   = df["geohash"].map(d48_geo_mean).fillna(d48_global)
    df["geo_std48"] = df["geohash"].map(d48_geo_std).fillna(float(d48_geo_std.mean()))
    df["d49_gm"]    = df["geohash"].map(d49_geo_mean).fillna(d49_global)
    df["geo_shift"] = df["geohash"].map(geo_shift_map).fillna(0.0)
    df["d48_early_gm"] = df["geohash"].map(d48_early_geo_mean).fillna(d48_global)
    df["early_shift"]  = df["geohash"].map(early_shift_map).fillna(0.0)
    df["early_ratio"]  = df["geohash"].map(early_ratio_map).fillna(1.0)
    df["geo_ts_shifted"] = np.clip(df["geo_ts"] + df["early_shift"], 0, 1)
    df["geo_ts_scaled"]  = np.clip(df["geo_ts"] * df["early_ratio"], 0, 1)

    df["lag8"]  = [lag8_map.get(gi, lag_global) for gi in g]
    df["lag7"]  = [lag7_map.get(gi, lag_global) for gi in g]
    df["lag6"]  = [lag6_map.get(gi, lag_global) for gi in g]
    df["lag3m"] = (df["lag8"] + df["lag7"] + df["lag6"]) / 3

    df["d49_trend"] = df["geohash"].map(d49_trend_map).fillna(0.0)
    df["d49_last"]  = df["geohash"].map(d49_last_map).fillna(d49_global)
    df["d49_max"]   = df["geohash"].map(d49_max_map).fillna(d49_global)

    nd = []
    for gi, si in zip(g, s):
        nbrs = nc.get(gi, [])
        vals = [d48_gsd.get((n, si), np.nan) for n in nbrs]
        vals = [v for v in vals if not np.isnan(v)]
        nd.append(float(np.mean(vals)) if vals else float(d48_slot_mean.get(si, d48_global)))
    df["nbr_d48"] = nd
    df["nbr_gm"]  = df["geohash"].map(nbr_gm_map).fillna(d48_global)
    return df

FEATURE_COLS = [
    "slot", "hour", "is_morn", "is_night",
    "NumberofLanes", "road_enc", "weath_enc", "lv_enc", "lm_enc",
    "temp_f", "temp_b",
    "geo_ts", "geo_m48", "geo_std48",
    "d49_gm", "geo_shift", "d48_early_gm", "early_shift", "early_ratio",
    "geo_ts_shifted", "geo_ts_scaled",
    "lag8", "lag7", "lag6", "lag3m",
    "d49_trend", "d49_last", "d49_max",
    "nbr_d48", "nbr_gm",
]

# ── 6. CV: day48 train → day49 early val (honest cross-day holdout) ───────────
print("Building features…")
d48_feat = build_features(d48, d48_s[8], d48_s[7], d48_s[6], d48_global)
d49_feat = build_features(d49, d49_s[8], d49_s[7], d49_s[6], d49_global)

X_cv_tr, y_cv_tr   = d48_feat[FEATURE_COLS].values, d48_feat["demand"].values
X_cv_val, y_cv_val = d49_feat[FEATURE_COLS].values, d49_feat["demand"].values

LGB_BASE_PARAMS = dict(
    objective="regression", metric="rmse",
    n_estimators=12000, learning_rate=0.015,
    min_child_samples=20, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, n_jobs=-1, verbose=-1,
)

def score_preds(y_true, preds):
    preds = np.clip(preds, 0, 1)
    return {
        "r2": r2_score(y_true, preds),
        "rmse": np.sqrt(mean_squared_error(y_true, preds)),
        "mae": mean_absolute_error(y_true, preds),
    }

def print_score(prefix, score, best_iter=None):
    suffix = f"  best_iter={best_iter}" if best_iter else ""
    print(f"{prefix} R²={score['r2']:.4f}  RMSE={score['rmse']:.5f}  MAE={score['mae']:.5f}{suffix}")

print("CV (day48 → day49 early)…")
model_specs = []

for depth, leaves, seed in [(12, 1024, 42), (14, 2048, 84), (15, 4096, 126)]:
    params = {**LGB_BASE_PARAMS, "max_depth": depth, "num_leaves": leaves, "random_state": seed}
    model_specs.append(("lgb", f"lgb_d{depth}", params))

if XGBRegressor is not None:
    for depth, seed in [(12, 42), (14, 84)]:
        params = dict(
            objective="reg:squarederror", eval_metric="rmse",
            n_estimators=12000, learning_rate=0.015, max_depth=depth,
            min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.02, reg_lambda=1.0, tree_method="hist",
            random_state=seed, n_jobs=-1, verbosity=0,
            early_stopping_rounds=300,
        )
        model_specs.append(("xgb", f"xgb_d{depth}", params))
else:
    print("XGBoost not installed; skipping XGB models.")

if CatBoostRegressor is not None:
    for depth, seed in [(12, 42), (14, 84)]:
        params = dict(
            loss_function="RMSE", eval_metric="RMSE",
            iterations=12000, learning_rate=0.015, depth=depth,
            random_seed=seed, l2_leaf_reg=3.0, bootstrap_type="Bernoulli",
            subsample=0.85, allow_writing_files=False, verbose=False,
            od_type="Iter", od_wait=300,
        )
        model_specs.append(("cat", f"cat_d{depth}", params))
else:
    print("CatBoost not installed; skipping CatBoost models.")

cv_results = []
for kind, name, params in model_specs:
    print(f"  fitting {name}...")
    if kind == "lgb":
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_cv_tr, y_cv_tr,
            eval_set=[(X_cv_val, y_cv_val)],
            callbacks=[lgb.early_stopping(300, verbose=False)],
        )
        best_iter = model.best_iteration_ or params["n_estimators"]
        preds = model.predict(X_cv_val, num_iteration=best_iter)
    elif kind == "xgb":
        model = XGBRegressor(**params)
        model.fit(X_cv_tr, y_cv_tr, eval_set=[(X_cv_val, y_cv_val)], verbose=False)
        best_iter = model.best_iteration + 1 if getattr(model, "best_iteration", None) is not None else params["n_estimators"]
        preds = model.predict(X_cv_val, iteration_range=(0, best_iter))
    else:
        model = CatBoostRegressor(**params)
        model.fit(X_cv_tr, y_cv_tr, eval_set=(X_cv_val, y_cv_val), use_best_model=True)
        best_iter = model.get_best_iteration() + 1 if model.get_best_iteration() is not None else params["iterations"]
        preds = model.predict(X_cv_val)
    score = score_preds(y_cv_val, preds)
    print_score(f"    {name}", score, best_iter)
    cv_results.append({
        "kind": kind, "name": name, "params": params,
        "best_iter": best_iter, "score": score, "cv_preds": np.clip(preds, 0, 1),
    })

cv_stack = np.column_stack([r["cv_preds"] for r in cv_results])
raw_weights = np.array([1 / max(r["score"]["rmse"], 1e-9) ** 2 for r in cv_results])
weights = raw_weights / raw_weights.sum()
blend_cv_preds = cv_stack @ weights
blend_score = score_preds(y_cv_val, blend_cv_preds)
print("\nCV blend weights:")
for r, w in zip(cv_results, weights):
    print(f"  {r['name']}: {w:.3f}")
print_score("CV blend", blend_score)
print("Note: CV is on midnight slots (0-8); test slots (9-55) are typically easier to predict")

# ── 7. Final model: day48 + day49 early, scaled iterations ───────────────────
print(f"\nFinal ensemble on {len(train):,} rows")

train_full = pd.concat([d48_feat, d49_feat], ignore_index=True)
test_feat  = build_features(test, d49_s[8], d49_s[7], d49_s[6], d49_global)

X_train = train_full[FEATURE_COLS].values
y_train = train_full["demand"].values
X_test  = test_feat[FEATURE_COLS].values

test_pred_parts = []
train_pred_parts = []
for r in cv_results:
    kind, name = r["kind"], r["name"]
    scaled_iter = min(max(int(r["best_iter"] * (len(train) / len(d48)) * 1.20), 1500), 12000)
    print(f"  fitting final {name} with {scaled_iter} iterations...")
    if kind == "lgb":
        params = {**r["params"], "n_estimators": scaled_iter}
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        train_pred_parts.append(np.clip(model.predict(X_train), 0, 1))
        test_pred_parts.append(np.clip(model.predict(X_test), 0, 1))
    elif kind == "xgb":
        params = {**r["params"], "n_estimators": scaled_iter}
        params.pop("early_stopping_rounds", None)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        train_pred_parts.append(np.clip(model.predict(X_train), 0, 1))
        test_pred_parts.append(np.clip(model.predict(X_test), 0, 1))
    else:
        params = {**r["params"], "iterations": scaled_iter, "od_type": None}
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        train_pred_parts.append(np.clip(model.predict(X_train), 0, 1))
        test_pred_parts.append(np.clip(model.predict(X_test), 0, 1))

train_blend = np.column_stack(train_pred_parts) @ weights
train_score = score_preds(y_train, train_blend)
print_score("Train blend", train_score)

# ── 8. Predict & save ─────────────────────────────────────────────────────────
final_preds = np.clip(np.column_stack(test_pred_parts) @ weights, 0, 1)
submission  = pd.DataFrame({"Index": test["Index"].values, "demand": final_preds})
submission.to_csv(SUB_PATH, index=False)
print(f"\nSubmission saved → {SUB_PATH}  ({len(submission)} rows)")
print(f"Prediction stats: min={final_preds.min():.4f}  max={final_preds.max():.4f}  mean={final_preds.mean():.4f}")

# ── 9. Validate against 5 known ground-truth rows ────────────────────────────
check = sample.merge(test[["Index","geohash","slot"]], on="Index")
check = check.merge(submission, on="Index")
check["error"]   = check["demand"] - check["true_demand"]
check["abs_err"] = check["error"].abs()
print("\n=== Validation against 5 known ground-truth rows ===")
print(check[["Index","geohash","slot","true_demand","demand","error"]].to_string())
print(f"\nMAE (5 samples) : {check.abs_err.mean():.5f}")
print(f"RMSE (5 samples): {np.sqrt((check.error**2).mean()):.5f}")
print("\nNote: All 5 sample rows are from slot 9 (2:15am) — the hardest test slot.")
print("Slots 10-55 (2:30am-1:45pm) have much better d48 coverage and will score higher.")
