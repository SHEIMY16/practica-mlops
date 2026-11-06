"""Etapa evaluate: mide el modelo en el conjunto de prueba y escribe metrics.json."""
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

target = yaml.safe_load(open("params.yaml", encoding="utf-8"))["prepare"]["target"]

test = pd.read_csv("data/prepared/test.csv")
X, y = test.drop(columns=[target]), test[target]
pred = joblib.load("models/model.pkl").predict(X)

metrics = {
    "r2": round(r2_score(y, pred), 4),
    "rmse": round(root_mean_squared_error(y, pred), 4),
    "mae": round(mean_absolute_error(y, pred), 4)
}
Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
