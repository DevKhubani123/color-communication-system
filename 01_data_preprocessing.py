import pandas as pd

df = pd.read_csv("merged_trainval.csv")

# Drop junk index column if present
for col in ["Unnamed: 0", "index"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Spectral columns
r_cols = [c for c in df.columns if c.startswith('R') and c[1:].isdigit()]
r_cols = sorted(r_cols, key=lambda x: int(x[1:]))

# B-band columns
b_cols = [
 'B352S','B610','B616','B622','B623','B625','B628','B641',
 'B642','B648','B649','B651','B653','B662','B664','B671','B673'
]
b_cols = [c for c in b_cols if c in df.columns]

# Lab & RGB targets (adjust names if slightly different)
lab_cols = [c for c in df.columns if 'L_D65' in c or 'a_D65' in c or 'b_D65' in c]
rgb_cols = ['R', 'G', 'B']

print("R:", len(r_cols), "B:", len(b_cols), "Lab:", lab_cols, "RGB:", rgb_cols)
