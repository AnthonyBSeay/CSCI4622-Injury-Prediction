# 🏀 NBA Injury Prediction Using Machine Learning

---

## 📌 Problem Objective and Overview

This project builds machine learning models to predict lower-body non-contact injuries in NBA players using game logs, player tracking data, and injury history.

The goal is to identify patterns related to player workload, fatigue, and performance that may increase injury risk. Multiple classification models are trained and evaluated to estimate injury probability.

NBA injury prediction is a highly imbalanced, rare-event classification problem where only a small fraction of player-game samples result in injuries. This makes traditional accuracy misleading and requires evaluation using recall, precision-recall AUC, and threshold-tuned decision making

This creates two major challenges:

- Standard accuracy is misleading (a model can appear “accurate” by predicting no injuries)
- The real objective is **early detection of injury risk**, not overall correctness

As a result, this project prioritizes:
- Recall (injury detection rate)
- Precision-Recall AUC (PR-AUC)
- Threshold-optimized decision making

This project is intended for data science applications in sports analytics and predictive modeling.

---

📊 Data

The dataset combines:

NBA player tracking data (via NBA API)
Historical injury data from Kaggle
https://www.kaggle.com/datasets/loganlauton/nba-injury-stats-1951-2023
⚠️ Important

A processed, model-ready dataset is included in:

data/processed/
No additional preprocessing is required to run the pipeline

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/AnthonyBSeay/CSCI4622-Injury-Prediction
cd CSCI4622-Injury-Prediction
```

### 2. Create Virtual Environment (Python 3.10)
``` bash
python3.10 -m venv test_env
```

### 3. Activate Environment
Windows (Powershell):
``` bash
test_env\Scripts\activate
```
Mac/Linux:
``` bash
source test_env/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

`main.py` contains 4 statements to control what models are trained and evaluated at run time, and multiple statements to control what graphs are displayed at runtime. Only edit these CONFIG variables before running the full pipeline

``` python
# -----------------------------
# CONFIG (EDIT ONLY THIS)
# -----------------------------
RUN_LOGISTIC = True   # train and evaluate logistic model
PLOT_LOGISTIC = False  # display individual logistic metric graphs

RUN_XGB = True                  # train and evaluate XGB modelx
RUN_LGB = True                  # train and evaluate LGB model
PLOT_GRADIENTBOOSTING = False    # display XGB and LGB metric graphs

RUN_RNN = True   # train and evaluate RNN model
epochs = 25     # number of epochs for RNN training
PLOT_RNN = False  # display RNN metric graphs

PLOT_ALL = True     # display comparison metric graphs for all models (requires >1 models)
```

Too run the full pipeline, run the following command in a virtual environment after installing the required dependencies.
```bash
python main.py
```

### Runtime
Logistic / XGBoost / LightGBM: ~1–2 minutes
RNN (25 epochs): ~5–10 minutes (CPU dependent)

## ✨ Features
- Injury prediction using classification models
- Feature engineering from player workload and tracking data
- Multiple model comparison (Logistic Regression, Gradient Boosting, RNN)
- Evaluation using ROC-AUC, precision, recall, and F1-score
- Visualization of model performance metrics

## 🧠 Tech Stack / Built With
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost / Gradient Boosting
- TensorFlow / PyTorch (RNN modeling)
- Matplotlib / Seaborn
- Jupyter Notebooks

## 📁 Directory Structure

``` bash
|-- main.py             # Full pipeline execution
|-- README.md           # Project documentation
|-- requirements.txt    # Dependencies
|-- NBA_Lower_Body_Injury_Prediction_Report.pdf   # Final project report deliverable
|-- notebooks           # EDA and experimentation
  |-- analysis.ipynb
  |-- RNN.ipynb
  |-- wrangle.ipynb
|-- data               
  |-- processed         # Processed datasets
  |-- raw               # Raw datasets
|-- results             # Outputs and evaluation
  |-- metrics.ipynb
  |-- graphs
|-- src           # Source code
  |-- util.py             
  |-- models            # Final Modulized models
    |-- gradient_boosting.py  
    |-- logistic.py
    |-- RNN.py
    |-- __pycache__
|-- .git
|-- .gitattributes
|-- .gitignore
|-- .vscode             # Editor configuration
```

## Result Summary

| Model                       | Accuracy | Precision | **Recall** | F1 Score | **PR-AUC** | Interpretation                                                                                    |
| --------------------------- | -------- | --------- | ---------- | -------- | ------ | ------------------------------------------------------------------------------------------------- |
| Logistic Regression         | 0.44     | 0.02      | **0.98**   | 0.04     | **0.105**  | Extreme over-prediction; captures nearly all injuries but unusable due to massive false positives |
| Gradient Boosting (XGBoost) | 0.72     | 0.03      | **0.77**   | 0.06     | **0.043**  | Better balance than logistic but still poor precision; struggles with class imbalance             |
| LightGBM                    | 0.74     | 0.03      | **0.74**   | 0.06     | **0.045**  | Slightly more stable than XGBoost; best traditional ML model but still weak signal                |
| RNN (LSTM)                  | 0.82     | 0.18      | **0.62**   | 0.28     | **0.198**  | Strongest overall model; captures temporal patterns and significantly improves precision          |


- Injury prediction is fundamentally a **rare-event learning problem**
- Traditional ML models struggle with precision-recall tradeoffs
- Logistic regression achieves high recall but produces many false positives
- Gradient boosting models improve accuracy but fail to capture temporal dynamics
- The LSTM model significantly improves predictive performance by modeling player fatigue over time

A detailed technical report is included:

`NBA_Lower_Body_Injury_Prediction_Report.pdf`

---

## License

This project is for educational purposes. You may reuse or modify this code with proper attribution.
