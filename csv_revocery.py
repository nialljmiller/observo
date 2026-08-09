import json
import csv

# Paths
input_file = 'all_data.json'
output_file = 'all_data.csv'

# Load JSON
with open(input_file, 'r') as f:
    data = json.load(f)

# Make CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

print(f"Done. Wrote {len(data)} rows to {output_file}.")

