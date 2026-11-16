
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "converted_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "frontier_cleaned_model.joblib"

print("Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. TARGET
# --------------------------------------------------

df["energy_kwh"] = (
    pd.to_numeric(df["mean_power"], errors="coerce")
    * pd.to_numeric(df["run_time"], errors="coerce")
) / 3_600_000

# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------

df["submit_time"] = pd.to_datetime(
    df["submit_time"], utc=True, errors="coerce"
)

df["submit_hour"] = df["submit_time"].dt.hour
df["submit_dayofweek"] = df["submit_time"].dt.dayofweek
df["submit_month"] = df["submit_time"].dt.month

df["is_weekend"] = df["submit_dayofweek"].isin([5, 6]).astype(int)

df["hour_sin"] = np.sin(
    2 * np.pi * df["submit_hour"] / 24
)
df["hour_cos"] = np.cos(
    2 * np.pi * df["submit_hour"] / 24
)

df["node_group"] = pd.cut(
    df["node_count"],
    bins=[0, 1, 4, 16, 64, 256, float("inf")],
    labels=[
        "1 Node", "2-4 Nodes", "5-16 Nodes",
        "17-64 Nodes", "65-256 Nodes", "257+ Nodes"
    ]
)

df["wall_time_group"] = pd.cut(
    df["wall_time"],
    bins=[0, 10, 30, 60, 120, 240, 1440],
    labels=[
        "1-10 min", "11-30 min", "31-60 min",
        "61-120 min", "121-240 min", "241+ min"
    ],
    include_lowest=True
)

# --------------------------------------------------
# 4. FEATURES
# --------------------------------------------------

features = [
    "node_count", "wall_time", "queue", "job_mode",
    "gpu_enabled", "interconnect_exists",
    "submit_hour", "submit_dayofweek", "submit_month",
    "is_weekend", "hour_sin", "hour_cos",
    "node_group", "wall_time_group", "project_prefix"
]

# Remove invalid target/timestamp rows
work = df.dropna(
    subset=["energy_kwh", "submit_time"]
).copy()

work = work[
    np.isfinite(work["energy_kwh"]) &
    (work["energy_kwh"] > 0)
].sort_values("submit_time").reset_index(drop=True)

work["end_time"] = pd.to_datetime(
    work["end_time"], utc=True, errors="coerce"
)

# --------------------------------------------------
# 5. CHRONOLOGICAL SPLIT
# --------------------------------------------------

n = len(work)

i_train = int(n * 0.70)
i_val = int(n * 0.85)

validation_start = work.loc[i_train, "submit_time"]

train_mask = (
    (work.index < i_train) &
    (work["end_time"] < validation_start)
)

val_mask = (
    (work.index >= i_train) &
    (work.index < i_val)
)

test_mask = work.index >= i_val

X = work[features]
y = work["energy_kwh"]

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

if X_train.empty or X_test.empty:
    raise ValueError("Training or testing data is empty.")

print("\nChronological split:")
print("Training:", len(X_train))
print("Validation:", int(val_mask.sum()))
print("Testing:", len(X_test))

# --------------------------------------------------
# 6. PREPROCESSING
# --------------------------------------------------

numeric_features = [
    "node_count", "wall_time", "submit_hour",
    "submit_dayofweek", "submit_month",
    "is_weekend", "hour_sin", "hour_cos"
]

categorical_features = [
    "queue", "job_mode", "gpu_enabled",
    "interconnect_exists", "node_group",
    "wall_time_group", "project_prefix"
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# --------------------------------------------------
# 7. RANDOM FOREST MODEL
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining cleaned-data Random Forest...")
model.fit(X_train, y_train)

# --------------------------------------------------
# 8. EVALUATION
# --------------------------------------------------

print("\nGenerating test predictions...")
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== CLEANED MODEL RESULTS ==========")
print(f"MAE:  {mae:.3f} kWh")
print(f"RMSE: {rmse:.3f} kWh")
print(f"R2:   {r2:.4f}")

# --------------------------------------------------
# 9. SAVE MODEL AND RESULTS
# --------------------------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

results = pd.DataFrame({
    "actual_energy_kwh": y_test.values,
    "predicted_energy_kwh": predictions
})

results["absolute_error_kwh"] = (
    results["actual_energy_kwh"] -
    results["predicted_energy_kwh"]
).abs()

results.to_csv(
    BASE_DIR / "cleaned_validation_results.csv",
    index=False
)

print("\nCleaned model saved:", MODEL_PATH)
print("Predictions saved: cleaned_validation_results.csv")
