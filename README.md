# ML_Projects

# 🚀 End-to-End MLOps Pipeline with CI/CD using GitHub Actions, Docker & MLflow

# STEP 1: Setup Project Structure
# ├── data/               # Raw datasets
# ├── models/             # Saved models
# ├── src/                # Source code
# │   ├── train.py        # Training script
# │   └── utils.py        # Helper functions
# ├── requirements.txt    # Python dependencies
# ├── Dockerfile          # Docker container config
# ├── .github/workflows/
# │   └── mlops.yml       # GitHub Actions CI/CD pipeline
# ├── mlruns/             # MLflow tracking
# ├── README.md

# =========================
# STEP 2: training script (src/train.py)
# =========================

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/iris.csv")
X = data.drop("species", axis=1)
y = data["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")
    print(f"Logged model with accuracy: {acc}")

# =========================
# STEP 3: requirements.txt
# =========================

mlflow
scikit-learn
pandas

# =========================
# STEP 4: Dockerfile
# =========================

FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "src/train.py"]

# =========================
# STEP 5: GitHub Actions Workflow (.github/workflows/mlops.yml)
# =========================

name: MLOps CI/CD

on:
  push:
    branches: [ main ]

jobs:
  train-model:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Repo
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Train model
      run: |
        python src/train.py

# =========================
# STEP 6: Dataset (data/iris.csv)
# =========================
# You can download from:
# https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv

# =========================
# STEP 7: Run Locally
# =========================
# 1. Run training manually:
#    python src/train.py
# 2. Use Docker:
#    docker build -t mlops-mlflow .
#    docker run mlops-mlflow

# =========================
# STEP 8: Push to GitHub
# =========================
# git init
# git add .
# git commit -m "Initial commit for MLOps pipeline"
# git remote add origin <your_repo_url>
# git push -u origin main

# =========================
# STEP 9: Optional - Deploy model
# =========================
# You can serve the model with MLflow:
# mlflow models serve -m mlruns/<run_id>/artifacts/model -p 1234
