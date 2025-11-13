import pandas as pd
import numpy as np
import os

print("Current working directory:", os.getcwd())

if not os.path.exists("merged_trainval.csv"):
    print("ERROR: merged_trainval.csv not found here!")
    raise SystemExit

# Load CSV
df = pd.read_csv("merged_trainval.csv")

# Remove junk columns
for col in ["Unnamed: 0", "index"]:
    if col in df.columns:
        df = df.drop(columns=[col])
        print("Dropped:", col)

print("\nAll columns in CSV:")
print(df.columns.tolist())
print()

# ---- AUTO-DETECT LAB COLUMNS ----
# Strategy:
# pick columns that:
# - contain 'D65' (illumination)
# - and start with 'L', 'a', or 'b' (case-insensitive)
lab_candidates = []
for c in df.columns:
    lc = c.lower()
    if 'd65' in lc and lc[0] in ['l', 'a', 'b']:
        lab_candidates.append(c)

print("Auto-detected Lab candidates:", lab_candidates)

if len(lab_candidates) == 3:
    lab_cols = lab_candidates
else:
    # If auto-detect fails, STOP and let you choose manually
    print("\nERROR: Could not reliably detect exactly 3 Lab columns.")
    print("Look at the column list above and identify the 3 Lab columns (L*, a*, b*).")
    print("Then set lab_cols = ['exact_L_name', 'exact_a_name', 'exact_b_name'] manually.")
    raise SystemExit

print("Using Lab columns:", lab_cols)

# Extract Lab values
lab_values = df[lab_cols].values

# Get IDs
id_col = None
for possible_id in ["ImitationID", "ID", "SampleID"]:
    if possible_id in df.columns:
        id_col = possible_id
        break

if id_col:
    ids = df[id_col].values
    print("Using ID column:", id_col)
else:
    ids = np.arange(len(df))
    print("No ID column found, using numeric IDs 0..N-1")

# Save
np.save("lab_values.npy", lab_values)
np.save("ids.npy", ids)

print("\nSaved lab_values.npy and ids.npy successfully ✅")
print("lab_values shape:", lab_values.shape)
print("ids shape:", ids.shape)

