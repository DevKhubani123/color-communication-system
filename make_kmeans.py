import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

print("Current working directory:", os.getcwd())
print("Files in this directory:", os.listdir())

# 1. Check CSV
if not os.path.exists("merged_trainval.csv"):
    print("ERROR: merged_trainval.csv not found in this folder.")
    raise SystemExit

# 2. Load CSV
df = pd.read_csv("merged_trainval.csv")
print("Loaded CSV with shape:", df.shape)

# 3. Drop junk columns
for col in ["Unnamed: 0", "index"]:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"Dropped column: {col}")

# 4. Detect Lab columns (print all to be safe)
print("All columns:", df.columns.tolist())

# 👉 If your exact Lab names differ, update this list manually:
possible_lab = ['L_D65-10°', 'a_D65-10°', 'b_D65-10°', 'L_D65-10Â°', 'a_D65-10Â°', 'b_D65-10Â°']
lab_cols = [c for c in df.columns if c in possible_lab]

print("Detected Lab columns:", lab_cols)

if len(lab_cols) != 3:
    print("ERROR: Lab columns not detected correctly. Fix lab_cols list in script.")
    raise SystemExit

X_lab = df[lab_cols].values
print("X_lab shape:", X_lab.shape)

# 5. Train KMeans
k = 8
print(f"Training KMeans with k={k}...")
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_lab)
print("KMeans training done.")

# 6. Save model
joblib.dump(kmeans, "kmeans_lab.pkl")
print("Saved: kmeans_lab.pkl")

# 7. Save clustered CSV (optional)
df['cluster'] = kmeans.labels_
df.to_csv("with_clusters.csv", index=False)
print("Saved: with_clusters.csv")
