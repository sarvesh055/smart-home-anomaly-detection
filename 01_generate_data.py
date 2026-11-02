import pandas as pd
import numpy as np
import datetime

# Set random seed so you get the exact same results every time
np.random.seed(42)

# We will simulate 60 days of data
days = 60
start_date = datetime.datetime(2024, 1, 1)

data = []

print("Simulating smart home sensor data...")

for day in range(days):
    current_date = start_date + datetime.timedelta(days=day)
    
    # ---------------------------------------------------------
    # SIMULATE NORMAL BEHAVIOR
    # ---------------------------------------------------------
    # 1. Bedroom (Wake up between 6:30 AM and 8:00 AM)
    wake_hour = np.random.normal(7, 0.5) # Mean 7 AM, std 0.5 hr
    wake_time = current_date + datetime.timedelta(hours=wake_hour)
    
    # 2. Bathroom (Morning routine, 7:15 AM to 7:45 AM)
    bath_time = wake_time + datetime.timedelta(minutes=np.random.randint(15, 45))
    
    # 3. Kitchen (Breakfast, 8:00 AM to 9:00 AM)
    kitchen_time = wake_time + datetime.timedelta(hours=1, minutes=np.random.randint(0, 60))
    
    # 4. Bedroom (Going to sleep between 9:30 PM and 11:00 PM)
    sleep_hour = np.random.normal(22.5, 0.5)
    sleep_time = current_date + datetime.timedelta(hours=sleep_hour)

    daily_events = [
        {"timestamp": wake_time, "sensor": "Bedroom", "status": "ON"},
        {"timestamp": bath_time, "sensor": "Bathroom", "status": "ON"},
        {"timestamp": kitchen_time, "sensor": "Kitchen", "status": "ON"},
        {"timestamp": sleep_time, "sensor": "Bedroom", "status": "ON"}
    ]

    # ---------------------------------------------------------
    # INJECT ANOMALIES (The fun part!)
    # ---------------------------------------------------------
    # Anomaly 1: The "Fall" (Days 25 to 27). Person stays in bed.
    if 25 <= day <= 27:
        daily_events = [{"timestamp": sleep_time, "sensor": "Bedroom", "status": "ON"}]
        
    # Anomaly 2: Night Wandering (Day 45). Person goes to kitchen at 2 AM.
    if day == 45:
        wander_time = current_date + datetime.timedelta(hours=2, minutes=30)
        daily_events.append({"timestamp": wander_time, "sensor": "Kitchen", "status": "ON"})

    data.extend(daily_events)

# Convert to DataFrame
df = pd.DataFrame(data)
df = df.sort_values(by="timestamp").reset_index(drop=True)

# Save to CSV
df.to_csv("sensor_data.csv", index=False)

print(f"Successfully generated {len(df)} sensor events across {days} days.")
print("Anomalies injected: 'Fall' (Days 25-27), 'Night Wandering' (Day 45)")
print("\nHere is what the first 5 rows look like:")
print(df.head())