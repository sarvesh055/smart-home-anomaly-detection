import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv("sensor_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Feature Engineering: Summarize by Day
df['date'] = df['timestamp'].dt.date

# Count triggers per day
daily_features = df.groupby(['date', 'sensor']).size().unstack(fill_value=0).reset_index()

# NEW: Calculate how many morning bathroom visits occurred
def count_morning_bathroom(group):
    return len(group[(group['sensor'] == 'Bathroom') & (group['timestamp'].dt.hour < 10)])

morning_baths = df.groupby('date').apply(count_morning_bathroom).reset_index(name='Morning_Bathroom_Visits')

# Merge this new feature in
daily_features = daily_features.merge(morning_baths, on='date')

# Calculate total daily activity
daily_features['Total_Activity'] = daily_features[['Bedroom', 'Bathroom', 'Kitchen']].sum(axis=1)

# 3. Train the Isolation Forest
# Using ALL 5 features, including the morning bathroom visits
model_features = ['Bedroom', 'Bathroom', 'Kitchen', 'Total_Activity', 'Morning_Bathroom_Visits']
X = daily_features[model_features]

# Set contamination to 0.08 (roughly 5 anomalies out of 60 days)
model = IsolationForest(contamination=0.08, random_state=42)
daily_features['Anomaly'] = model.fit_predict(X)

# Map to readable labels
daily_features['Anomaly_Label'] = daily_features['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# 4. Print the days the model flagged
print("\n--- ANOMALIES DETECTED ---")
anomalies = daily_features[daily_features['Anomaly_Label'] == 'Anomaly']
print(anomalies[['date', 'Total_Activity', 'Bedroom', 'Bathroom', 'Kitchen', 'Morning_Bathroom_Visits']])

# 5. Visualize the results
plt.figure(figsize=(12, 5))
sns.scatterplot(
    data=daily_features, 
    x='date', 
    y='Total_Activity', 
    hue='Anomaly_Label', 
    palette={'Normal': 'blue', 'Anomaly': 'red'},
    s=100
)

plt.title("Anomaly Detection: Daily Smart Home Activity")
plt.xlabel("Date")
plt.ylabel("Total Sensor Triggers (Daily)")
plt.xticks(rotation=45, ha='right')
plt.gca().set_xticks(daily_features['date'][::5]) 
plt.tight_layout()
plt.savefig("anomaly_detection_results.png")
plt.show()