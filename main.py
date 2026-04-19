# main.py

from src.models.logistic import train_logistic_model_from_csv, plot_logistic_results
from src.models.gradient_boosting import gradient_boosting_pipeline, plot_gradient_boosting_results
from src.models.RNN import train_rnn_from_csv, plot_rnn_results
import pandas as pd


# -----------------------------
# CONFIG (EDIT ONLY THIS)
# -----------------------------
RUN_LOGISTIC = True
RUN_XGB = True
RUN_LGB = True
RUN_RNN = False


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


# =============================
# 1 Logistic Regression
# =============================
if RUN_LOGISTIC:
    print("\n================ LOGISTIC REGRESSION ================\n")

    results_logistic = train_logistic_model_from_csv(
        csv_path="data/model_ready_data.csv",
        target_col="injury",
        feature_cols=feature_cols,
        threshold=0.05,
        verbose=True
    )

    plot_logistic_results(results_logistic)


# =============================
# 2 Gradient Boosting
# =============================
if RUN_XGB or RUN_LGB:
    print("\n================ GRADIENT BOOSTING ================\n")

    df = pd.read_csv("data/model_ready_data.csv")
    X = df[feature_cols]
    y = df["injury"]

    model_type = "both"
    if RUN_XGB and not RUN_LGB:
        model_type = "xgb"
    elif RUN_LGB and not RUN_XGB:
        model_type = "lgb"

    results_gb = gradient_boosting_pipeline(
        X, y,
        model_type=model_type,
        threshold=0.05
    )

    plot_gradient_boosting_results(results_gb)


# =============================
# 3 RNN (optional)
# =============================
if RUN_RNN:
    print("\n================ RNN MODEL ================\n")

    rnn_results = train_rnn_from_csv(
        csv_path="data/model_ready_data.csv",
        feature_cols=feature_cols,
        target_col="injury",
        sequence_length=5,
        epochs=20
    )

    plot_rnn_results(rnn_results)