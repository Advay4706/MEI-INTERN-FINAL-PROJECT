import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("graphs", exist_ok=True)

master_df = pd.read_csv("data/processed/master_dataset.csv")
temp_df = pd.read_csv("data/processed/temperature_readings.csv")

master_df["DateTime"] = pd.to_datetime(
    master_df["Date"].astype(str) + " " + master_df["Time"],
    format="%Y%m%d %H:%M:%S"
)

start_time = master_df["DateTime"].min()

master_df["ElapsedTime"] = (
    master_df["DateTime"] - start_time
).dt.total_seconds() / 60

temp_df["Time"] = pd.to_datetime(
    temp_df["Time"],
    format="%H:%M"
)

temp_df["ElapsedTime"] = (
    temp_df["Time"] - temp_df["Time"].iloc[0]
).dt.total_seconds() / 60

print("\n================ DATASET INFORMATION ================\n")

print(master_df.info())

print("\n=====================================================\n")

print("\n================ SUMMARY STATISTICS =================\n")

print(master_df.describe())

print("\n=====================================================\n")

print("\n================ MISSING VALUES =====================\n")

print(master_df.isnull().sum())

print("\n=====================================================\n")

plt.figure(figsize=(12,6))

plt.plot(
    temp_df["ElapsedTime"],
    temp_df["SpindleTemp"],
    marker="o",
    linewidth=2
)

plt.title(
    "Spindle Temperature vs Elapsed Time",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Elapsed Time (Minutes)",
    fontsize=12
)

plt.ylabel(
    "Spindle Temperature (°C)",
    fontsize=12
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "graphs/01_spindle_temperature.png",
    dpi=300
)

plt.close()

plt.figure(figsize=(12,6))

plt.plot(
    temp_df["ElapsedTime"],
    temp_df["ZTemp"],
    marker="o",
    linewidth=2
)

plt.title(
    "Z-Axis Temperature vs Elapsed Time",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Elapsed Time (Minutes)",
    fontsize=12
)

plt.ylabel(
    "Z-Axis Temperature (°C)",
    fontsize=12
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "graphs/02_z_temperature.png",
    dpi=300
)

plt.close()

plt.figure(figsize=(12,6))

plt.plot(
    master_df["ElapsedTime"],
    master_df["Z_Error"],
    linewidth=1.8
)

plt.title(
    "Z-Axis Position Error vs Elapsed Time",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Elapsed Time (Minutes)",
    fontsize=12
)

plt.ylabel(
    "Z Position Error (mm)",
    fontsize=12
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "graphs/03_z_error.png",
    dpi=300
)

plt.close()
plt.figure(figsize=(12,6))

plt.scatter(
    master_df["SpindleTemp"],
    master_df["Z_Error"],
    s=20
)

plt.title(
    "Spindle Temperature vs Z-Axis Position Error",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Spindle Temperature (°C)",
    fontsize=12
)

plt.ylabel(
    "Z Position Error (mm)",
    fontsize=12
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "graphs/04_spindle_temp_vs_z_error.png",
    dpi=300
)

plt.close()


plt.figure(figsize=(12,6))

plt.scatter(
    master_df["ZTemp"],
    master_df["Z_Error"],
    s=20
)

plt.title(
    "Z-Axis Temperature vs Z-Axis Position Error",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Z-Axis Temperature (°C)",
    fontsize=12
)

plt.ylabel(
    "Z Position Error (mm)",
    fontsize=12
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "graphs/05_z_temp_vs_z_error.png",
    dpi=300
)

plt.close()


correlation = master_df[
    [
        "RPM",
        "SpindleTemp",
        "XTemp",
        "YTemp",
        "ZTemp",
        "X_Error",
        "Y_Error",
        "Z_Error"
    ]
].corr()

plt.figure(figsize=(10,8))

image = plt.imshow(
    correlation,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(image)

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title(
    "Correlation Heatmap",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "graphs/06_correlation_heatmap.png",
    dpi=300
)

plt.close()


rpm_summary = master_df.groupby("RPM")[
    [
        "SpindleTemp",
        "ZTemp",
        "Z_Error"
    ]
].mean()

plt.figure(figsize=(9,6))

rpm_summary.plot(
    kind="bar"
)

rpm_summary = master_df.groupby("RPM")[
    [
        "SpindleTemp",
        "ZTemp",
        "Z_Error"
    ]
].mean()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(
    rpm_summary.index.astype(str),
    rpm_summary["SpindleTemp"]
)

axes[0].set_title("Average Spindle Temperature")
axes[0].set_xlabel("RPM")
axes[0].set_ylabel("Temperature (°C)")
axes[0].grid(True, axis="y")

axes[1].bar(
    rpm_summary.index.astype(str),
    rpm_summary["ZTemp"]
)

axes[1].set_title("Average Z-Axis Temperature")
axes[1].set_xlabel("RPM")
axes[1].set_ylabel("Temperature (°C)")
axes[1].grid(True, axis="y")

axes[2].bar(
    rpm_summary.index.astype(str),
    rpm_summary["Z_Error"]
)

axes[2].set_title("Average Z-Axis Position Error")
axes[2].set_xlabel("RPM")
axes[2].set_ylabel("Position Error (mm)")
axes[2].grid(True, axis="y")

plt.suptitle(
    "Comparison of Average Values at 6000 RPM and 12000 RPM",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "graphs/07_rpm_comparison.png",
    dpi=300
)

plt.close()


correlation.to_csv(
    "data/processed/correlation_matrix.csv"
)

print("\n================ CORRELATION MATRIX ================\n")

print(correlation)

print("\n====================================================")

print("\nEDA COMPLETED SUCCESSFULLY!")

print("\nGenerated Graphs:")

print("1. Spindle Temperature vs Time")
print("2. Z Temperature vs Time")
print("3. Z Position Error vs Time")
print("4. Spindle Temperature vs Z Error")
print("5. Z Temperature vs Z Error")
print("6. Correlation Heatmap")
print("7. RPM Comparison")

print("\nCorrelation matrix saved successfully.")