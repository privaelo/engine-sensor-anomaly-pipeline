
# Engine Sensor Anomaly Detection

An end-to-end machine learning pipeline for detecting anomalies in multivariate vehicle sensor data using Isolation Forest and Airflow orchestration.

## Project Overview

This project simulates a predictive maintenance system using:
- NASA CMAPSS turbofan engine dataset
- Sliding window time-series feature generation
- Unsupervised anomaly detection
- Airflow DAGs for repeatable ML workflows
- Jupyter-based visual dashboards

## Tech Stack

- Python, Pandas, NumPy, Scikit-learn
- Airflow (via Docker)
- Matplotlib, Seaborn
- Isolation Forest (unsupervised ML)

## Dashboard Samples

### Sensor 2 Trend with Anomalies (Engine 3)
![Sensor Trend](outputs/sensor_trend_anomalies.png)

### Anomaly Count per Engine
![Anomaly Bar Chart](outputs/anomaly_counts_per_engine.png)

### Sensor Correlation Heatmap
![Heatmap](outputs/sensor_correlation_heatmap.png)

## How to Run

1. Clone the repo:
```bash
git clone https://github.com/
cd engine-sensor-anomaly-pipeline
```

2. Set up Airflow with Docker:
```bash
cd docker-compose-airflow
docker-compose up airflow-init
docker-compose up
```

3. Add your DAG to `dags/` and run it via the Airflow UI.

## Future Enhancements

- FastAPI scoring microservice
- Slack/email anomaly alerts
- Retraining DAG
- Streamlit dashboard

## Folder Structure

```
engine-sensor-anomaly-pipeline/
├── README.md
├── requirements.txt
├── docker-compose-airflow/
│   └── docker-compose.yaml
├── dags/
│   ├── score_pipeline.py
│   ├── features.py
├── data/
│   └── processed/
├── models/
│   └── isolation_forest.pkl
├── outputs/
│   ├── anomaly_alerts.csv
│   ├── sensor_trend_anomalies.png
│   ├── anomaly_counts_per_engine.png
│   └── sensor_correlation_heatmap.png
├── notebooks/
│   └── anomaly_dashboard.ipynb
```
