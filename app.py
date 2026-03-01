# streamlit_manual_predict.py
import streamlit as st
import numpy as np
import joblib

# --- Load your saved model and encoders ---
model = joblib.load("C:/Users/MSI/Desktop/PFE/PHASE 2/script/miracle_ai_model.pkl")
encoders = joblib.load("C:/Users/MSI/Desktop/PFE/PHASE 2/script/encoders.pkl")

# Columns in your model
feature_columns = ['microrna', 'microrna_group_simplified', 'parasite', 'organism', 'infection', 'cell type', 'time']

st.title("Manual microRNA Prediction")
st.write("Predict whether a miRNA is UP or DOWN-regulated based on its features.")

# --- Input fields ---
manual_data = {}
manual_data['microrna'] = st.text_input("miRNA", "hsa-let-7a")
manual_data['microrna_group_simplified'] = st.text_input("miRNA Family", "let-7a")
manual_data['parasite'] = st.text_input("Parasite", "L.major")
manual_data['organism'] = st.text_input("Organism", "Human")
manual_data['infection'] = st.text_input("Infection", "in vitro")
manual_data['cell type'] = st.text_input("Cell Type", "PBMC")
manual_data['time'] = st.number_input("Time (hours)", min_value=0, max_value=100, value=3)

# --- Encode manual row ---
encoded_input = []
unseen_cols = []

for col in feature_columns:
    val = manual_data[col]
    if col in encoders:
        if isinstance(val, str):
            val = val.strip()  # clean spaces
        try:
            encoded_val = encoders[col].transform([val])[0]
        except:
            encoded_val = 0  # fallback for unseen values
            unseen_cols.append(col)
        encoded_input.append(encoded_val)
    else:
        encoded_input.append(val)  # numeric like 'time'

encoded_input = np.array([encoded_input])

# --- Predict ---
prediction = model.predict(encoded_input)[0]
prob = model.predict_proba(encoded_input)[0][prediction]

# --- Decode numeric features back to text for display ---
decoded_info = {}
for col in feature_columns:
    if col in encoders:
        decoded_info[col] = encoders[col].inverse_transform([encoded_input[0][feature_columns.index(col)]])[0]
    else:
        decoded_info[col] = manual_data[col]

# --- Display results ---
st.subheader("Input Data")
for col in feature_columns:
    st.write(f"• {col}: {decoded_info[col]}")

st.subheader("Prediction")
st.write(f"• Prediction: {'UP' if prediction==1 else 'DOWN'}")
st.write(f"• Confidence: {prob*100:.1f}%")

if unseen_cols:
    st.warning(f"⚠ The following columns had values never seen in training: {', '.join(unseen_cols)}\nPrediction may be less reliable for these features.")
