
# Engine Sensor Anomaly Detection

An end-to-end machine learning pipeline for detecting anomalies in multivariate vehicle sensor data using Isolation Forest and Airflow orchestration.

## Project Overview

This project simulates a predictive maintenance system using:
- NASA CMAPSS turbofan engine dataset
- Sliding window time-series feature generation
- Unsupervised anomaly detection
- Airflow DAGs for repeatable ML workflows
- Jupyter-based visual dashboards

## Requirements

- Python, Pandas, NumPy, Scikit-learn
- Airflow (via Docker)
- Matplotlib, Seaborn
- Isolation Forest (unsupervised ML)

## Dashboard Samples

### Anomaly Score Distribution
<img width="1204" height="928" alt="anomaly_distribution" src="https://github.com/user-attachments/assets/377f742b-7f70-488a-8eb1-d47eb5bd369a" />


### Anomaly Count per Engine
<img width="1280" height="683" alt="anomaly_per_engine" src="https://github.com/user-attachments/assets/5166491f-c40e-46c0-8d45-52919d3e4b11" />

### Sensor Correlation Heatmap
![Heatmap](<img width="2400" height="2000" alt="sensor_correlation_heatmap" src="https://github.com/user-attachments/assets/fe510af3-7729-4eb7-b335-5278b6d4b7e5" />
)

## How to Run

1. Clone the repo:
```bash
git clone https://github.com/privaelo/engine-sensor-anomaly-detection.git
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

<!-- ## Folder Structure

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
```-->
