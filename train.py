import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# =====================
# 1. Load the Iris dataset
# =====================
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

# =====================
# 2. Train/Test Split
# =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# =====================
# 3. Train Model
# =====================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# =====================
# 4. Log Experiment with MLflow
# =====================
mlflow.set_experiment("iris_classification")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    # Log model with signature & input example
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_test.iloc[:1],
        signature=infer_signature(X_test, preds)
    )

# =====================
# 5. Output Accuracy
# =====================
print(f"✅ Model trained with accuracy: {acc:.4f}")
