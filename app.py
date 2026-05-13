import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ── Model Definition (must match your notebook exactly) ──────────────────────
class Powerplant(nn.Module):
    def __init__(self):
        super(Powerplant, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, x):
        return self.model(x)

# ── Load model (cached so it loads only once) ─────────────────────────────────
@st.cache_resource
def load_model():
    model = Powerplant()
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# ── Fit scaler on training data (same split as notebook) ──────────────────────
@st.cache_resource
def load_scaler():
    df = pd.read_csv("powerplant_data.csv")
    X = df.drop("PE", axis=1)
    from sklearn.model_selection import train_test_split
    X_train, _, _, _ = train_test_split(X, df["PE"], test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Power Plant Output Predictor", page_icon="⚡")
st.title("⚡ Power Plant Energy Output Predictor")
st.markdown("Enter the environmental conditions to predict the **net electrical energy output (PE)** in MW.")

model  = load_model()
scaler = load_scaler()

col1, col2 = st.columns(2)

with col1:
    AT = st.number_input("🌡️ Ambient Temperature (AT) °C", min_value=0.0,  max_value=100.0,  value=20.0, step=0.1)
    V  = st.number_input("💨 Exhaust Vacuum (V) cm Hg",    min_value=25.0, max_value=200.0,  value=50.0, step=0.1)

with col2:
    AP = st.number_input("🔵 Ambient Pressure (AP) mbar",  min_value=990.0, max_value=2035.0, value=1010.0, step=0.1)
    RH = st.number_input("💧 Relative Humidity (RH) %",    min_value=25.0, max_value=100.0,  value=70.0,  step=0.1)

if st.button("🔮 Predict Power Output", use_container_width=True):
    input_data   = np.array([[AT, V, AP, RH]])
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled).float()

    with torch.no_grad():
        prediction = model(input_tensor).item()

    st.success(f"### Predicted Net Power Output: **{prediction:.2f} MW**")
    st.balloons()
