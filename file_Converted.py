import pandas as pd

data = {
    "Requirement_1": [
        "The system must achieve a top speed of 80 km/h under standard conditions (rider weight 75 kg).",
        "The motor power output must not exceed 750W under normal operation.",
        "The system must prioritize battery efficiency over performance when battery level drops below 20%.",
        "The system must support a range of 100 km on a single charge at 40 km/h.",
        "The vehicle must operate silently below 20 km/h for urban environments.",
        "The safety system must lock the vehicle if the rider’s helmet is not detected.",
        "The system must reduce power output by 20% in eco mode.",
        "The vehicle must withstand 1000 charge cycles without battery capacity dropping below 80%.",
        "The system must limit speed to 25 km/h in pedestrian zones.",
        "The dashboard must dedicate 80% of the screen to navigation during active routes."
    ],
    "Requirement_2": [
        "The system must limit acceleration to 0.5 m/s² when carrying a passenger (total weight > 120 kg).",
        "The vehicle must accelerate from 0 to 50 km/h in under 6 seconds.",
        "The vehicle must maintain stability at speeds up to 80 km/h on wet surfaces.",
        "The battery must maintain 90% capacity after 500 cycles at 25°C.",
        "The horn must produce a sound level of at least 90 dB at 5 meters.",
        "The system must allow manual override of all safety locks via a key fob.",
        "The traction control system must activate within 50 ms of wheel slip detection.",
        "The battery must charge from 0% to 80% within 4 hours using a standard charger.",
        "The vehicle must accelerate from 0 to 50 km/h in under 6 seconds.",
        "The dashboard must show real-time diagnostics using at least 50% of the screen."
    ]
}

df = pd.DataFrame(data)
df.to_excel("test_data.xlsx", index=False)
print("test_data.xlsx has been created.")

df = pd.read_excel("test_data.xlsx")
df.to_csv("test_data.csv", index=False)
print("Converted to test_data.csv")