# main.py

from src.models.logistic import train_logistic_model_from_csv, plot_logistic_results
from src.models.gradient_boosting import gradient_boosting_pipeline
from src.models.RNN import train_rnn_from_csv
import pandas as pd

# -----------------------------
# 1 Define Features
# -----------------------------
feature_cols = [
    "minutes_float", "distance", "speed",
    "distance_per_min", "speed_per_min",
    "rolling_distance_3", "rolling_speed_3", "rolling_minutes_3",
    "rolling_distance_5", "rolling_speed_5", "rolling_minutes_5",
    "rolling_distance_10", "rolling_speed_10", "rolling_minutes_10",
    "cumulative_distance", "cumulative_minutes",
    "rolling_distance_20", "acwr_distance",
    "distance_pct_change", "speed_pct_change",
    "distance_std_5", "distance_z", "speed_z",
    "minutes_prop", "load_intensity",
    "rolling_3d_minutes_avg", "rolling_3d_distance_avg", "rolling_3d_speed_avg",
    "rolling_3d_load_sum", "rolling_7d_minutes_avg", "rolling_7d_distance_avg",
    "rolling_7d_speed_avg", "rolling_7d_load_sum"
]

# -----------------------------
# 2 Logistic Regression
# -----------------------------
results_logistic = train_logistic_model_from_csv(
    csv_path="data/model_ready_data.csv",
    target_col="injury",
    feature_cols=feature_cols,
    threshold=0.05,
    verbose=True
)

#plot_logistic_results(results_logistic)

# -----------------------------
# 3 Gradient Boosting (XGBoost + LightGBM)
# -----------------------------
#Load CSV manually for gradient boosting
# df = pd.read_csv("data/model_ready_data.csv")
# X = df[feature_cols]
# y = df["injury"]

# print("\n=== Gradient Boosting Models ===")
# results_gb = gradient_boosting_pipeline(X, y, model_type="both", threshold=0.05)

# # Access metrics for XGBoost
# if "xgb" in results_gb:
#     print("\n--- XGBoost ---")
#     print("Top 10 features:\n", results_gb["xgb"]["feature_importances"].head(10))
#     print("ROC-AUC:", results_gb["xgb"]["model"].score(X, y))

# # Access metrics for LightGBM
# if "lgb" in results_gb:
#     print("\n--- LightGBM ---")
#     print("Top 10 features:\n", results_gb["lgb"]["feature_importances"].head(10))
#     print("ROC-AUC:", results_gb["lgb"]["model"].score(X, y))

# -----------------------------
# 4 LSTM
# -----------------------------
# print("\n=== RNN Model ===")

# rnn_results = train_rnn_from_csv(
#     csv_path="data/model_ready_data.csv",
#     feature_cols=feature_cols,
#     target_col="injury",
#     sequence_length=5,
#     epochs=25
# )

