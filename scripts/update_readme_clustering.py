import sys

with open('/hermes_workspace/bnpl_ml_project/README.md', 'r') as f:
    lines = f.readlines()

# Find start and end indices
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if line.strip() == '### 4. Customer Segmentation Analysis (Clustering)':
        start_idx = i
    if line.strip() == '## Dashboard' and start_idx is not None:
        end_idx = i
        break

if start_idx is None or end_idx is None:
    print("Could not find markers")
    sys.exit(1)

# Read CLUSTERING_REPORT.md and skip first line
with open('/hermes_workspace/bnpl_ml_project/CLUSTERING_REPORT.md', 'r') as f:
    cluster_lines = f.readlines()[1:]  # skip the first line (title)

# Build new lines: keep everything up to start_idx, then the header line, then cluster_lines, then everything from end_idx onward
new_lines = lines[:start_idx+1] + cluster_lines + lines[end_idx:]

with open('/hermes_workspace/bnpl_ml_project/README.md', 'w') as f:
    f.writelines(new_lines)

print("README.md updated with clustering report content.")