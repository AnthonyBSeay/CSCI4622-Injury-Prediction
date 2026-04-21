# main.py

from src.models.logistic import train_logistic_model_from_csv, plot_logistic_results
from src.models.gradient_boosting import gradient_boosting_pipeline, plot_gradient_boosting_results
from src.models.RNN import train_rnn_from_csv, plot_rnn_results
from src.util import compare_all_models
import pandas as pd


# -----------------------------
# CONFIG (EDIT ONLY THIS)
# -----------------------------
RUN_LOGISTIC = True   # train and evaluate logistic model
PLOT_LOGISTIC = False  # display individual logistic metric graphs

RUN_XGB = True                  # train and evaluate XGB modelx
RUN_LGB = True                  # train and evaluate LGB model
PLOT_GRADIENTBOOSTING = False    # display XGB and LGB metric graphs

RUN_RNN = True   # train and evaluate RNN model
epochs = 25      # number of epochs for RNN training
PLOT_RNN = False  # display RNN metric graphs

PLOT_ALL = True     # display comparison metric graphs for all models (requires >1 models)

# ---------

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

if PLOT_LOGISTIC:
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

if PLOT_GRADIENTBOOSTING:
    plot_gradient_boosting_results(results_gb)


# =============================
# 3 RNN 
# =============================
if RUN_RNN:
    print("\n================ RNN MODEL ================\n")

    rnn_results = train_rnn_from_csv(
        csv_path="data/model_ready_data.csv",
        feature_cols=feature_cols,
        target_col="injury",
        sequence_length=5,
        epochs=epochs
    )

if PLOT_RNN:
    plot_rnn_results(rnn_results)


if PLOT_ALL:
    # =============================
    # Collect Model Outputs
    # =============================
    model_outputs = []

    # ---- Logistic Regression ----
    if RUN_LOGISTIC:
        model_outputs.append({
            "name": "Logistic Regression",
            "y_true": results_logistic["y_test"],
            "y_pred": results_logistic["y_pred"],
            "y_proba": results_logistic["y_probs"]
        })


    # ---- Gradient Boosting ----
    if RUN_XGB or RUN_LGB:

    # XGBoost
        if RUN_XGB and "xgb" in results_gb:
            model_outputs.append({
                "name": "XGBoost",
                "y_true": results_gb["xgb"]["y_test"],
                "y_pred": results_gb["xgb"]["y_pred"],
                "y_proba": results_gb["xgb"]["y_probs"]
            })

        # LightGBM
        if RUN_LGB and "lgb" in results_gb:
            model_outputs.append({
                "name": "LightGBM",
                "y_true": results_gb["lgb"]["y_test"],
                "y_pred": results_gb["lgb"]["y_pred"],
                "y_proba": results_gb["lgb"]["y_probs"]
            })


    # ---- RNN (LSTM) ----
    if RUN_RNN:
        # Need to convert probabilities -> predictions using threshold
        y_pred_rnn = (rnn_results["y_probs"] >= rnn_results["threshold"]).astype(int)

        model_outputs.append({
            "name": "RNN (LSTM)",
            "y_true": rnn_results["y_true"],
            "y_pred": y_pred_rnn,
            "y_proba": rnn_results["y_probs"]
        })


    # =============================
    # Compare All Models
    # =============================
    if len(model_outputs) > 1:
        compare_all_models(model_outputs)
    else:
        print("⚠️ Not enough models to compare.")