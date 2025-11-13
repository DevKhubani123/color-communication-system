import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from utils_color import deltaE76

# Optional: RGB->Lab using skimage if available
try:
    from skimage.color import rgb2lab
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# ============ HELPERS ============

def load_dataset(path="merged_trainval.csv"):
    if not os.path.exists(path):
        st.error(f"Dataset '{path}' not found in this folder.")
        st.stop()
    df = pd.read_csv(path)
    for col in ["Unnamed: 0", "index"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def load_forward_model(df):
    """
    Tries with-B model first, else no-B.
    Returns (model, X_scaler, y_scaler, feature_cols, mode_str)
    """
    # With B-bands
    if os.path.exists("model_forward_withB.pkl"):
        model = joblib.load("model_forward_withB.pkl")
        Xs = joblib.load("Xs_withB.pkl")
        ys = joblib.load("ys_withB.pkl")
        feature_cols = joblib.load("feature_cols_withB.pkl")
        return model, Xs, ys, feature_cols, "withB"

    # Fallback: no-B model
    if os.path.exists("model_forward_noB.pkl"):
        model = joblib.load("model_forward_noB.pkl")
        Xs = joblib.load("Xs_noB.pkl")
        ys = joblib.load("ys_noB.pkl")
        # derive r_cols from df
        r_cols = sorted([c for c in df.columns if c.startswith('R') and c[1:].isdigit()],
                        key=lambda x: int(x[1:]))
        feature_cols = r_cols
        return model, Xs, ys, feature_cols, "noB"

    return None, None, None, None, None

def load_kmeans():
    if os.path.exists("kmeans_lab.pkl"):
        return joblib.load("kmeans_lab.pkl")
    return None

def load_lab_values_ids():
    if os.path.exists("lab_values.npy") and os.path.exists("ids.npy"):
        lab_vals = np.load("lab_values.npy")
        ids = np.load("ids.npy", allow_pickle=True)
        return lab_vals, ids
    return None, None

def load_reverse_model():
    if os.path.exists("model_reverse.pkl"):
        model = joblib.load("model_reverse.pkl")
        Xs = joblib.load("Xs_rev.pkl")
        ys = joblib.load("ys_rev.pkl")
        return model, Xs, ys
    return None, None, None

def rgb_to_hex(rgb):
    r, g, b = [int(np.clip(x, 0, 255)) for x in rgb]
    return '#%02x%02x%02x' % (r, g, b)

def safe_rgb_from_row(row, rgb_cols):
    if all(c in row.index for c in rgb_cols):
        return [row[c] for c in rgb_cols]
    return [128, 128, 128]

def get_lab_cols(df):
    # auto-detect 3 Lab columns
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if "d65" in lc and lc[0] in ["l", "a", "b"]:
            candidates.append(c)
    if len(candidates) == 3:
        return candidates
    # fallback: manual (edit if needed)
    manual = ['L_D65-10Â°', 'a_D65-10Â°', 'b_D65-10Â°']
    if all(c in df.columns for c in manual):
        return manual
    st.error("Could not detect Lab columns. Please fix get_lab_cols() in app.py.")
    st.stop()

def nn_color_match(input_lab, lab_values, ids, top_n=5):
    """Return top_n nearest samples (id, Lab, distance)."""
    diffs = lab_values - input_lab
    dists = np.linalg.norm(diffs, axis=1)
    idxs = np.argsort(dists)[:top_n]
    return [(ids[i], lab_values[i], dists[i]) for i in idxs]

# ============ LOAD DATA & MODELS ============

st.set_page_config(page_title="Color Communication System", layout="wide")

df = load_dataset()
lab_cols = get_lab_cols(df)
rgb_cols = ['R', 'G', 'B']

fwd_model, Xs_fwd, ys_fwd, feature_cols, fwd_mode = load_forward_model(df)
kmeans = load_kmeans()
lab_values, ids = load_lab_values_ids()
rev_model, Xs_rev, ys_rev = load_reverse_model()

# ============ SIDEBAR ============

st.sidebar.title("Input / Options")
st.sidebar.success(f"Loaded dataset: merged_trainval.csv ({len(df)} rows)")

input_mode = st.sidebar.radio(
    "Choose input mode",
    ["Pick sample from dataset", "Custom RGB (color matching)", "About / Help"]
)

if fwd_model is None:
    st.sidebar.error("Forward model not found. Train & save model_forward_withB.pkl or model_forward_noB.pkl.")
else:
    st.sidebar.info(f"Using forward model: {fwd_mode}")

if kmeans is None:
    st.sidebar.warning("kmeans_lab.pkl not found → cluster view limited.")
if lab_values is None:
    st.sidebar.warning("lab_values.npy / ids.npy not found → color matching limited.")
if not SKIMAGE_AVAILABLE:
    st.sidebar.warning("Install 'scikit-image' for accurate RGB→Lab in color matching (pip install scikit-image).")

# ============ MAIN TABS ============

tab1, tab2, tab3, tab4 = st.tabs(
    ["📘 Sample Prediction", "🎯 Color Matching", "🌈 Clusters", "🔁 Reverse Model "]
)

# ---------- TAB 1: SAMPLE PREDICTION ----------
with tab1:
    st.subheader("Sample-based Color Prediction")

    if fwd_model is None:
        st.info("Train and save the forward model to use this tab.")
    else:
        idx = st.slider("Pick sample index (0-based)", 0, len(df)-1, 0)
        sample = df.iloc[idx]
        st.info(f"Using sample index {idx} (ImitationID: {sample.get('ImitationID', 'N/A')})")

        # Show sample row table (compact)
        with st.expander("Show raw sample features"):
            st.dataframe(sample.to_frame().T)

        # Prepare input for model
        X = sample[feature_cols].values.reshape(1, -1)
        X_scaled = Xs_fwd.transform(X)
        y_pred_scaled = fwd_model.predict(X_scaled)
        y_pred = ys_fwd.inverse_transform(y_pred_scaled)[0]

        pred_lab = y_pred[:3]
        pred_rgb = np.clip(y_pred[3:], 0, 255)

        true_lab = sample[lab_cols].values
        true_rgb = safe_rgb_from_row(sample, rgb_cols)

        dE = deltaE76(true_lab, pred_lab)

        st.markdown("### Color Comparison")
        c1, c2 = st.columns(2)

        with c1:
            st.write("**Predicted color**")
            st.color_picker(
                "Predicted",
                rgb_to_hex(pred_rgb),
                key=f"pred_{idx}"
            )
            st.write("Predicted Lab:", [round(x, 2) for x in pred_lab])
            st.write("Predicted RGB:", [int(round(x)) for x in pred_rgb])

        with c2:
            st.write("**Actual color**")
            st.color_picker(
                "Actual",
                rgb_to_hex(true_rgb),
                key=f"true_{idx}"
            )
            st.write("Actual Lab:", [round(float(x), 2) for x in true_lab])
            st.write("Actual RGB:", [int(x) for x in true_rgb])

        st.markdown("### Metrics for this sample")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("ΔE76 (Pred vs Actual)", f"{dE:.2f}")
        with m2:
            diff_rgb = np.mean((np.array(true_rgb) - pred_rgb) ** 2) ** 0.5
            st.metric("RGB RMSE (this sample)", f"{diff_rgb:.2f}")

# ---------- TAB 2: COLOR MATCHING ----------
with tab2:
    st.subheader("Color Matching using Dataset Library")

    if lab_values is None:
        st.info("Run make_lab_values.py to enable this feature.")
    else:
        hex_color = st.color_picker("Pick a color to match", "#66cc66")
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)

        if SKIMAGE_AVAILABLE:
            # skimage expects RGB in [0,1]
            lab = rgb2lab(np.array([[[r/255, g/255, b/255]]]))[0,0,:]
            input_lab = lab
        else:
            st.warning("Using approximate Lab: install scikit-image for proper conversion.")
            # crude approximation: treat RGB as Lab-like
            input_lab = np.array([r/2.55, g-128, b-128])

        st.write("Input RGB:", (r, g, b))
        st.write("Input Lab (approx.):", [round(float(x), 2) for x in input_lab])

        matches = nn_color_match(input_lab, lab_values, ids, top_n=5)

        st.markdown("### Top 5 closest colors in dataset")
        for mid, mlab, dist in matches:
            row = df[df.get("ImitationID", pd.Series()) == mid]
            if len(row) == 0:
                # fallback if ImitationID not unique or missing
                rgb_guess = [128, 128, 128]
            else:
                row = row.iloc[0]
                rgb_guess = safe_rgb_from_row(row, rgb_cols)

            mhex = rgb_to_hex(rgb_guess)
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                st.color_picker("", mhex, key=f"match_{mid}", label_visibility="collapsed")
            with col2:
                st.write(f"ID: {mid}")
                st.write("Lab:", [round(float(x), 2) for x in mlab])
            with col3:
                st.write(f"Distance ΔE (approx): {dist:.2f}")

# ---------- TAB 3: CLUSTERS ----------
with tab3:
    st.subheader("Color Clusters (K-Means on Lab Space)")

    if kmeans is None:
        st.info("Run make_kmeans.py to enable clusters.")
    else:
        centers = kmeans.cluster_centers_
        st.write(f"Number of clusters: {len(centers)}")

        st.markdown("### Cluster Centers")
        cols = st.columns(len(centers))
        for i, center in enumerate(centers):
            # convert Lab center to a fake RGB for swatch (no exact conv without skimage, keep simple)
            if SKIMAGE_AVAILABLE:
                # build dummy 1x1 Lab image
                from skimage.color import lab2rgb
                lab_ = np.array([[[center[0], center[1], center[2]]]])
                rgb_ = np.clip(lab2rgb(lab_), 0, 1)[0,0,:] * 255
            else:
                # simple hack: use L as gray base plus a/b tints
                L, a, b = center
                r = np.clip(L*2.55 + a, 0, 255)
                g = np.clip(L*2.55 - (a+b)/2, 0, 255)
                c = np.clip(L*2.55 + b, 0, 255)
                rgb_ = np.array([r, g, c])
            hex_ = rgb_to_hex(rgb_)
            with cols[i]:
                st.markdown(f"**C{i}**")
                st.color_picker("", hex_, key=f"c_center_{i}", label_visibility="collapsed")

        with st.expander("Show few samples from a selected cluster"):
            cid = st.number_input("Cluster ID", min_value=0, max_value=len(centers)-1, value=0)
            # Need cluster column; if not present, recompute quickly
            if "cluster" in df.columns:
                df_c = df[df["cluster"] == cid].head(12)
            else:
                # assign using kmeans
                df_c = df.copy()
                df_c["cluster"] = kmeans.predict(df[lab_cols].values)
                df_c = df_c[df_c["cluster"] == cid].head(12)

            st.write(f"Showing up to 12 samples from cluster {cid}:")
            for _, row in df_c.iterrows():
                rgb = safe_rgb_from_row(row, rgb_cols)
                hex_ = rgb_to_hex(rgb)
                c1, c2, c3 = st.columns([1, 2, 4])
                with c1:
                    st.color_picker("", hex_, key=f"cl_{cid}_{row.name}", label_visibility="collapsed")
                with c2:
                    st.write(f"ID: {row.get('ImitationID', row.name)}")
                with c3:
                    st.write("Lab:", [round(float(x), 2) for x in row[lab_cols].values])

# ---------- TAB 4: REVERSE MODEL ----------
with tab4:
    st.subheader("Reverse Model: Predict Spectrum from Lab ")

    if rev_model is None:
        st.info("Reverse model not found.  Train and save model_reverse.pkl, Xs_rev.pkl, ys_rev.pkl.")
    else:
        # Choose any sample to visualize
        idx_r = st.slider("Pick sample index for reverse model", 0, len(df)-1, 0, key="rev_idx")
        row = df.iloc[idx_r]
        lab = row[lab_cols].values
        st.write("Input Lab:", [round(float(x), 2) for x in lab])

        # Predict reflectance curve
        x = lab.reshape(1, -1)
        x_s = Xs_rev.transform(x)
        y_pred_s = rev_model.predict(x_s)
        y_pred = ys_rev.inverse_transform(y_pred_s)[0]

        # Collect R columns in order for plotting
        r_cols = sorted([c for c in df.columns if c.startswith('R') and c[1:].isdigit()],
                        key=lambda x: int(x[1:]))
        if len(r_cols) == len(y_pred):
            import plotly.express as px
            wl = [int(c[1:]) for c in r_cols]
            spec_df = pd.DataFrame({"Wavelength": wl, "Predicted Reflectance": y_pred})
            st.line_chart(spec_df, x="Wavelength", y="Predicted Reflectance")
        else:
            st.write("Reverse model output size does not match R-columns; check training setup.")
