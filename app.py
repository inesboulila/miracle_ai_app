import streamlit as st
import joblib                # <-- replace pickle with joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder   # still needed for the class reference

# --- Load model ---
try:
    model = joblib.load("miracle_ai_model.pkl")   # if model was saved with joblib, otherwise keep pickle
except FileNotFoundError:
    st.error("miracle_ai_model.pkl not found!")
    st.stop()

# --- Load encoders with joblib ---
try:
    encoders = joblib.load("encoders.pkl")        # now loaded with joblib
    use_encoders = True
except FileNotFoundError:
    st.warning("encoders.pkl not found! Assuming all numeric features.")
    use_encoders = False

# --- Feature columns ---
text_columns = ['microrna', 'microrna_group_simplified', 'parasite', 'organism', 'infection', 'cell type']
numeric_columns = ['time']

st.title("AI Prediction App")
st.write("Enter the features to predict 'is_upregulated'")

# --- Input fields ---
inputs = {}
for col in text_columns:
    inputs[col] = st.text_input(f"{col}")
inputs['time'] = st.number_input("time", step=1, value=0)
# --- Prediction ---
if st.button("Predict"):
    data = []
    unseen_cols = []  # keep track of unseen values
    # Encode categorical features
    for col in text_columns:
        val = inputs[col]
        if use_encoders:
            try:
                encoded = encoders[col].transform([val])[0]
            except:
                encoded = 0  # fallback for unseen values
                unseen_cols.append(col)
            data.append(encoded)
        else:
            data.append(val)
    # Add numeric features
    data.append(inputs['time'])

    # Convert to 2D array for model
    data = np.array([data])

    # Predict
    prediction = model.predict(data)
    prob = model.predict_proba(data)[0][prediction[0]]

    # Show results
    st.success(f"Prediction (is_upregulated): {prediction[0]}")
    st.info(f"Confidence: {prob*100:.1f}%")
    
    if unseen_cols:
        st.warning(f"The following columns had values never seen in training: {', '.join(unseen_cols)}")
