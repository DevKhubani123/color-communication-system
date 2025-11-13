import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib

df = pd.read_csv("merged_trainval.csv")
for col in ["Unnamed: 0", "index"]:
    if col in df.columns:
        df = df.drop(columns=[col])

r_cols = sorted([c for c in df.columns if c.startswith('R') and c[1:].isdigit()],
                key=lambda x: int(x[1:]))
lab_cols = [c for c in df.columns if 'L_D65' in c or 'a_D65' in c or 'b_D65' in c]

X = df[lab_cols].values.astype(float)
y = df[r_cols].values.astype(float)

X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42)

Xs = StandardScaler()
ys = StandardScaler()
X_tr_s = Xs.fit_transform(X_tr)
X_te_s = Xs.transform(X_te)
y_tr_s = ys.fit_transform(y_tr)

mlp_rev = MLPRegressor(hidden_layer_sizes=(128,64),
                       activation='relu',
                       solver='adam',
                       max_iter=800,
                       random_state=42)
mlp_rev.fit(X_tr_s, y_tr_s)

joblib.dump(mlp_rev,"model_reverse.pkl")
joblib.dump(Xs,"Xs_rev.pkl")
joblib.dump(ys,"ys_rev.pkl")
