import os
import re
from pathlib import Path
import pandas as pd

# Directory containing the txt files
data_dir = Path("./")
file_patterns = ["prune2-tp1.txt", "prune2-tp2.txt", "prune3-tp2.txt", "prune4-tp2.txt",
                 "prune2-tp4.txt", "prune3-tp4.txt", "prune4-tp4.txt", "prune5-tp4.txt",
                 "prune6-tp4.txt", "prune7-tp4.txt", "prune8-tp4.txt"]

# Prepare the structure to store results
results = []

# Regular expressions to match lines
re_allreduce = re.compile(r"\[AllReduce\] Time taken: ([\d.]+) ms")
re_dispatch = re.compile(r"\[Dispatch\]: ([\d.]+) ms")
re_forwarding = re.compile(r"\[Layer Forwarding Time\] \(\d+, ([\d.]+)\)")

# Parse each file
for file_name in file_patterns:
    file_path = data_dir / file_name
    if not file_path.exists():
        continue
    if file_name == "prune2-tp1.txt":
        with open(file_path, "r") as f:
            lines = f.readlines()[9000:]
    else:
        with open(file_path, "r") as f:
            lines = f.readlines()[-10000:]

    allreduce_times = []
    dispatch_times = []
    forwarding_times = []

    for line in lines:
        if "AllReduce" in line:
            match = re_allreduce.search(line)
            if match:
                allreduce_times.append(float(match.group(1)))
        elif "Dispatch" in line:
            match = re_dispatch.search(line)
            if match:
                dispatch_times.append(float(match.group(1)))
        elif "Layer Forwarding Time" in line:
            match = re_forwarding.search(line)
            if match:
                forwarding_times.append(float(match.group(1)) * 1000)  # convert to ms for consistency

    def compute_stats(data):
        return {
            "count": len(data),
            "mean": sum(data)/len(data) if data else 0,
            "median": sorted(data)[len(data)//2] if data else 0,
            "p99": sorted(data)[int(len(data)*0.99)] if data else 0,
            "p95": sorted(data)[int(len(data)*0.95)] if data else 0,
            "p90": sorted(data)[int(len(data)*0.90)] if data else 0,
            "p80": sorted(data)[int(len(data)*0.80)] if data else 0,
            "p70": sorted(data)[int(len(data)*0.70)] if data else 0,
        }

    stats = {
        "file": file_name,
        "allreduce": compute_stats(allreduce_times),
        "dispatch": compute_stats(dispatch_times),
        "forwarding": compute_stats(forwarding_times),
    }

    results.append(stats)

# Create a structured DataFrame
rows = []
for res in results:
    file = res["file"]
    for key in ["allreduce", "dispatch", "forwarding"]:
        row = {"file": file, "type": key}
        row.update(res[key])
        rows.append(row)

df = pd.DataFrame(rows)
df = df.rename(columns={
    "count": "Count",
    "mean": "Mean (ms)",
    "median": "Median (ms)",
    "p99": "P99 (ms)",
    "p95": "P95 (ms)",
    "p90": "P90 (ms)",
    "p80": "P80 (ms)",
    "p70": "P70 (ms)"})
# save the DataFrame to a CSV file
output_file = data_dir / "inference_log_stats.csv"
df.to_csv(output_file, index=False)
print(f"Statistics saved to {output_file}")