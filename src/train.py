"""Etapa train: entrena, registra el run en MLflow y guarda models/model.pkl."""
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
target = params["prepare"]["target"]
p = params["train"]
cfg = params["mlflow"]

train = pd.read_csv("data/prepared/train.csv")
test = pd.read_csv("data/prepared/test.csv")
X_train, y_train = train.drop(columns=[target]), train[target]
X_test, y_test = test.drop(columns=[target]), test[target]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(cfg["experiment"])

with mlflow.start_run():
    mlflow.log_params(p)

    model = RandomForestClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        random_state=params["prepare"]["random_state"],
    )
    model.fit(X_train, y_train)

    mlflow.log_params(params)

    pred = model.predict(X_test)
    mlflow.log_metrics({
        "r2": r2_score(y_test, pred),
        "rmse": root_mean_squared_error(y_test, pred),
        "mae": mean_absolute_error(y_test, pred)
    })

    artifactDF = pd.DataFrame(data={
        "y_real": y_test,
        "pred": pred,
        "delta": abs(y_test - pred)
    })
    mlflow.log_table(data=artifactDF, artifact_file="test_predictions.json")

    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=cfg["model_name"],
        input_example=X_train.head(3),
        serialization_format="cloudpickle",
    )

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Run registrado en MLflow y modelo guardado en models/model.pkl")