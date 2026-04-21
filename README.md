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

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/nba-injury-prediction.git
cd nba-injury-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage
Run full pipeline
```bash
python main.py
```

``` python
# -----------------------------
# CONFIG (EDIT ONLY THIS)
# -----------------------------
RUN_LOGISTIC = False
RUN_XGB = True
RUN_LGB = True
RUN_RNN = False
```

`main.py` contains 4 statements too control what models are trained and evaluated at run time. 

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

```bash
.
|-- main.py                  # Full pipeline execution
|-- test.py                  # Testing / validation script
|-- requirements.txt         # Dependencies
|-- README.md                # Project documentation

|-- data/                    # Raw and processed datasets
|   |-- gamelogs.csv
|   |-- gamelog_with_injuries.csv
|   |-- injury_with_games.csv
|   |-- merged_engineered_features.csv
|   |-- merged_tracking_injury_features.csv
|   |-- model_ready_data.csv
|   |-- NBA_Injury_Stats.csv
|   |-- tracking_partial.csv

|-- notebooks/               # EDA and experimentation
|   |-- analysis.ipynb
|   |-- wrangle.ipynb
|   |-- RNN.ipynb

|-- src/                     # Source code
|   |-- data/
|   |   |-- wrangle.py
|   |
|   |-- models/
|       |-- logistic.py
|       |-- gradient_boosting.py
|       |-- RNN.py

|-- results/                 # Outputs and evaluation
|   |-- metrics.ipynb
|   |-- graphs/
|       |-- Logistic_Precision_Recall.png
|       |-- Logistic_ROC.png

|-- .vscode/                # Editor configuration
|-- .gitignore
|-- .gitattributes
```

``` bash
|-- main.py             # Full pipeline execution
|-- README.md           # Project documentation
|-- requirements.txt    # Dependencies
|-- notebooks           # EDA and experimentation
  |-- analysis.ipynb
  |-- RNN.ipynb
  |-- wrangle.ipynb
|-- data                # Raw and processed datasets
|-- results             # Outputs and evaluation
  |-- metrics.ipynb
  |-- graphs
|-- src           # Source code
  |-- util.py     
  |-- models            # Final Mmdulized models
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

---

## License

This project is for educational purposes. You may reuse or modify this code with proper attribution.
