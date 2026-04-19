# 🏀 NBA Injury Prediction Using Machine Learning

---

## 📌 Description / Overview

This project builds machine learning models to predict lower-body non-contact injuries in NBA players using game logs, player tracking data, and injury history.

The goal is to identify patterns related to player workload, fatigue, and performance that may increase injury risk. Multiple classification models are trained and evaluated to estimate injury probability.

This project is intended for data science applications in sports analytics and predictive modeling.

---

## 📊 Demo (Optional)

*(Optional: insert GIF, screenshot, or notebook visualization here)*

Example outputs include:
- ROC curves
- Precision-recall curves
- Feature importance plots

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

## Result Summary (FINISH THIS)

## License

This project is for educational purposes. You may reuse or modify this code with proper attribution.
