import pandas as pd
position_df = pd.read_csv(
    "data/processed/position_clean.csv"
)
temperature_df = pd.read_csv(
    "data/processed/temperature_readings.csv"
)
position_df.rename(
    columns={
        "X_Drift": "X_Error",
        "Y_Drift": "Y_Error",
        "Z_Drift": "Z_Error"
    },
    inplace=True
)
position_df["Time"] = pd.to_datetime(
    position_df["Time"],
    format="%H:%M:%S"
)
temperature_df["Time"] = pd.to_datetime(
    temperature_df["Time"],
    format="%H:%M"
)
merged_df = pd.merge_asof(
    position_df.sort_values("Time"),
    temperature_df.sort_values("Time"),
    on="Time",
    direction="backward"
)
merged_df["Time"] = merged_df["Time"].dt.strftime("%H:%M:%S")
merged_df = merged_df[
    [
        "Date",
        "Time",
        "RPM",
        "SpindleTemp",
        "XTemp",
        "YTemp",
        "ZTemp",
        "X_Error",
        "Y_Error",
        "Z_Error"
    ]
]
print(merged_df.head())
print("\nDataset Information\n")
print(merged_df.info())
print("\nTotal Records =", len(merged_df))
print("\nMissing Values\n")
print(merged_df.isnull().sum())
merged_df.to_csv(
    "data/processed/master_dataset.csv",
    index=False
)
print("\nMaster Dataset Saved Successfully!")