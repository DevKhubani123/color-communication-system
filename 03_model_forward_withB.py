import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv("merged_trainval.csv")
for col in ["Unnamed: 0", "index"]:
    if col in df.columns:
        df = df.drop(columns=[col])

r_cols = sorted([c for c in df.columns if c.startswith('R') and c[1:].isdigit()],
                key=lambda x: int(x[1:]))

b_cols = [
 'B352S','B610','B616','B622','B623','B625','B628','B641',
 'B642','B648','B649','B651','B653','B662','B664','B671','B673'
]
b_cols = [c for c in b_cols if c in df.columns]

lab_cols = [c for c in df.columns if 'L_D65' in c or 'a_D65' in c or 'b_D65' in c]
rgb_cols = ['R','G','B']

feature_cols = r_cols + b_cols
X = df[feature_cols].values.astype(float)
y = df[lab_cols + rgb_cols].values.astype(float)
target_names = lab_cols + rgb_cols

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

Xs = StandardScaler()
ys = StandardScaler()
X_tr_s = Xs.fit_transform(X_tr)
X_te_s = Xs.transform(X_te)
y_tr_s = ys.fit_transform(y_tr)

mlp = MLPRegressor(hidden_layer_sizes=(128,64),
                   activation='relu',
                   solver='adam',
                   max_iter=800,
                   random_state=42)
mlp.fit(X_tr_s, y_tr_s)

y_pred_s = mlp.predict(X_te_s)
y_pred = ys.inverse_transform(y_pred_s)

rmse = {}
for i, name in enumerate(target_names):
    rmse[name] = mean_squared_error(y_te[:,i], y_pred[:,i])**0.5

print("RMSE WITH B-bands:")
for k,v in rmse.items():
    print(k, ":", round(v,4))

joblib.dump(mlp, "model_forward_withB.pkl")
joblib.dump(Xs, "Xs_withB.pkl")
joblib.dump(ys, "ys_withB.pkl")
joblib.dump(feature_cols, "feature_cols_withB.pkl")
