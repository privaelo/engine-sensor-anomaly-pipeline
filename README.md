
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
![Sensor Trend](<img width="2400" height="800" alt="sensor_trend_anomalies" src="https://github.com/user-attachments/assets/a5414b7f-0a2f-4416-a0a6-bc790b170ada" />
)

### Anomaly Count per Engine
![Anomaly Bar Chart](<img width="2000" height="1200" alt="anomaly_counts_per_engine" src="https://github.com/user-attachments/assets/1c67371a-9c34-4e33-b035-6d21724dc267" />
)

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
