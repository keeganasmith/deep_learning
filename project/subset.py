import pandas as pd
import joblib

# Load the original DataFrame
input_file = "results_dataframe.pkl"
df = joblib.load(input_file)

# Take the first million rows
subset = df.iloc[:1_000_000]

# Save the subset using joblib
output_file = "results_subset_1M.joblib"
joblib.dump(subset, output_file)

print(f"Saved {len(subset)} rows to '{output_file}'")
