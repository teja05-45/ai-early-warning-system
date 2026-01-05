import pandas as pd
import numpy as np

np.random.seed(42)

# Configuration
NUM_TICKETS = 300
DAYS = 30

ticket_ids = [f"TKT_{i:04d}" for i in range(1, NUM_TICKETS + 1)]
dates = pd.date_range(end=pd.Timestamp.today(), periods=DAYS)

data = []

for ticket in ticket_ids:
    priority = np.random.choice(
        ["Low", "Medium", "High", "Critical"],
        p=[0.4, 0.3, 0.2, 0.1]
    )

    past_sla_breaches = np.random.randint(0, 4)
    reopen_count = np.random.randint(0, 3)

    base_resolution_time = np.random.uniform(1, 10)

    for day in dates:
        ticket_age = np.random.randint(1, 60)
        engineer_load = np.random.randint(1, 25)
        escalations = np.random.binomial(1, 0.1)
        weekend_flag = int(day.weekday() >= 5)

        workload_trend_7d = np.random.uniform(-0.5, 0.5)
        resolution_delay_trend = np.random.uniform(-0.3, 0.7)

        risk_score = (
            0.4 * (priority in ["High", "Critical"]) +
            0.3 * (engineer_load > 15) +
            0.2 * (resolution_delay_trend > 0.3) +
            0.1 * escalations
        )

        risk_label = int(risk_score > 0.6)

        data.append([
            ticket,
            day,
            ticket_age,
            priority,
            engineer_load,
            past_sla_breaches,
            base_resolution_time,
            reopen_count,
            escalations,
            weekend_flag,
            workload_trend_7d,
            resolution_delay_trend,
            risk_label
        ])

columns = [
    "ticket_id",
    "date",
    "ticket_age_days",
    "priority",
    "assigned_engineer_load",
    "past_sla_breaches",
    "avg_resolution_time",
    "reopen_count",
    "customer_escalations",
    "weekend_flag",
    "workload_trend_7d",
    "resolution_delay_trend",
    "risk_label"
]

df = pd.DataFrame(data, columns=columns)

# Save dataset
df.to_csv("data/raw/operational_risk_data.csv", index=False)

print("✅ Synthetic dataset generated successfully")
print(df.head())
