from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import joblib, numpy as np, pandas as pd
import os

# Load your window generator
from features import generate_windows

def load_data():
    global df
    df = pd.read_csv("dags/data/processed/cleaned_FD001.csv")
    df.to_csv("dags/data/tmp/cleaned_for_dag.csv", index=False)

def create_windows():
    df = pd.read_csv("dags/data/tmp/cleaned_for_dag.csv")
    features = [c for c in df.columns if 'sensor' in c or 'op_set' in c]
    global X, unit_ids, end_cycles
    features = [c for c in df.columns if 'sensor' in c or 'op_set' in c]
    X, _, unit_ids, end_cycles = generate_windows(df, features, window_size=30, stride=1)
    np.save("dags/data/tmp/X.npy", X)
    np.save("dags/data/tmp/unit_ids.npy", unit_ids)
    np.save("dags/data/tmp/end_cycles.npy", end_cycles)


def score_anomalies():
    model = joblib.load("dags/models/isolation_forest.pkl")  # safer path

    X = np.load("dags/data/tmp/X.npy")
    unit_ids = np.load("dags/data/tmp/unit_ids.npy")
    end_cycles = np.load("dags/data/tmp/end_cycles.npy")

    scores = -model.score_samples(X)
    thresh = np.percentile(scores, 98)

    alerts = pd.DataFrame({
        "unit": unit_ids,
        "cycle": end_cycles,
        "score": scores,
        "is_anomaly": scores >= thresh
    })

    os.makedirs("dags/outputs", exist_ok=True)
    alerts.to_csv("dags/outputs/anomaly_alerts.csv", index=False)

def notify_if_anomalies():
    alerts = pd.read_csv("dags/outputs/anomaly_alerts.csv")
    num_alerts = alerts['is_anomaly'].sum()
    
    if num_alerts > 0:
        with open("dags/outputs/alert_log.txt", "a") as f:
            f.write(f"[ALERT] {num_alerts} anomalies detected at {datetime.now()}\n")
    else:
        print("No anomalies — no alert generated.")


default_args = {
    'start_date': datetime(2023, 1, 1)
}

with DAG("score_vehicle_anomalies", schedule=None, default_args=default_args, catchup=False, tags=["ml"]) as dag:
    t1 = PythonOperator(task_id="load_data", python_callable=load_data)
    t2 = PythonOperator(task_id="generate_windows", python_callable=create_windows)
    t3 = PythonOperator(task_id="score_and_save", python_callable=score_anomalies)
    t4 = PythonOperator(task_id="notify_if_alerts", python_callable=notify_if_anomalies)


    t1 >> t2 >> t3 >> t4
# This DAG processes vehicle sensor data to detect anomalies using an Isolation Forest model.
# It loads data, generates sliding windows, scores anomalies, and logs alerts if any anomalies are detected.
