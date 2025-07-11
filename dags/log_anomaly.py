import pandas as pd
df = pd.read_csv("dags/outputs/anomaly_alerts.csv")
print(f"Total anomalies flagged: {df['is_anomaly'].sum()}")
print(df[df['is_anomaly'] == True].head())
