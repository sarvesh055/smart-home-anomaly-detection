import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("sensor_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract the hour of the day to see when things happen
df['hour'] = df['timestamp'].dt.hour

# Plot 1: When do they use the bathroom or kitchen?
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='hour', hue='sensor', palette=['#4C72B0', '#DD8452', '#55A868'])
plt.title("Sensor Triggers by Hour of the Day")
plt.xlabel("Hour of the Day (0 = Midnight, 12 = Noon)")
plt.ylabel("Number of Triggers")
plt.legend(title="Room")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("eda_hourly_pattern.png")
plt.show()

# Plot 2: A timeline of events over the 60 days to visually spot the anomalies
plt.figure(figsize=(12, 4))
sensors = {'Bedroom': 1, 'Bathroom': 2, 'Kitchen': 3}
y_values = df['sensor'].map(sensors)

plt.scatter(df['timestamp'], y_values, c=y_values, cmap='viridis', alpha=0.6)
plt.yticks([1, 2, 3], ['Bedroom', 'Bathroom', 'Kitchen'])
plt.title("60-Day Sensor Timeline (Can you spot the anomalies?)")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig("eda_timeline.png")
plt.show()