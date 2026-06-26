import pandas as pd

# Open the position log
with open("data/raw/position_log.txt", "r") as file:
    lines = file.readlines()

records = []
current_record = {}

for line in lines:

    line = line.strip()

    # New record starts
    if line == "START":
        current_record = {}

    # Date
    elif line.startswith("DATE"):
        current_record["Date"] = line.split()[1]

    # Time
    elif line.startswith("TIME"):
        time = line.split()[1]

        # Convert 103356 -> 10:33:56
        formatted_time = (
            time[:2] + ":" +
            time[2:4] + ":" +
            time[4:]
        )

        current_record["Time"] = formatted_time

    # X Axis
    elif line.startswith("X"):
        current_record["X_Drift"] = float(
            line.split()[1]
        )

    # Y Axis
    elif line.startswith("Y"):
        current_record["Y_Drift"] = float(
            line.split()[1]
        )

    # Z Axis
    elif line.startswith("Z"):
        current_record["Z_Drift"] = float(
            line.split()[1]
        )

    # Record completed
    elif line == "END":
        records.append(current_record)

# Convert into DataFrame
df = pd.DataFrame(records)

print(df.head())

print("\nTotal Records =", len(df))

# Save CSV
df.to_csv(
    "data/processed/position_clean.csv",
    index=False
)

print("\nCSV Saved Successfully!")


