from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import joblib, numpy as np, pandas as pd
import os

# Load your window generator
from src.features import generate_windows

def load_data():
    global df
    df = pd.read_csv("data/processed/cleaned_FD001.csv")

def create_windows():
    global X, unit_ids, end_cycles
    features = [c for c in df.columns if 'sensor' in c or 'op_set' in c]
    X, _, unit_ids, end_cycles = generate_windows(df, features, window_size=30, stride=1)

def score_anomalies():
    model = joblib.load("models/isolation_forest.pkl")
    scores = -model.score_samples(X)
    thresh = np.percentile(scores, 98)

    alerts = pd.DataFrame({
        "unit": unit_ids,
        "cycle": end_cycles,
        "score": scores,
        "is_anomaly": scores >= thresh
    })

    alerts.to_csv("outputs/anomaly_alerts.csv", index=False)

default_args = {
    'start_date': datetime(2023, 1, 1),
    'catchup': False
}

with DAG("score_vehicle_anomalies", schedule_interval=None, default_args=default_args, tags=["ml"]) as dag:
    t1 = PythonOperator(task_id="load_data", python_callable=load_data)
    t2 = PythonOperator(task_id="generate_windows", python_callable=create_windows)
    t3 = PythonOperator(task_id="score_and_save", python_callable=score_anomalies)

    t1 >> t2 >> t3
