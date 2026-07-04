"""
app.py
-------
Streamlit UI for the ML-based Disease Prediction System.

The user checks off symptoms they're experiencing; a trained Random Forest
model predicts the most likely disease(s) with probability scores, plus
general (non-prescriptive) precaution guidance.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from predictor import predict, get_symptom_list, get_model_metrics, get_top_symptoms
from generate_dataset import PRECAUTIONS

st.set_page_config(page_title="Disease Prediction System", layout="wide")

st.title("🩺 ML-Based Disease Prediction System")
st.caption(
    "Random Forest classifier trained on symptom patterns across 20 common "
    "conditions. Select your symptoms below to see the most likely matches."
)

st.warning(
    "⚠️ **Educational project — not a medical device.** This tool is trained on "
    "a small synthetic dataset for demonstration purposes and must not be used "
    "for real diagnosis. Always consult a licensed healthcare professional.",
    icon="⚠️",
)

metrics = get_model_metrics()
col1, col2, col3 = st.columns(3)
col1.metric("Test Accuracy", f"{metrics['test_accuracy']*100:.1f}%")
col2.metric("5-Fold CV Accuracy", f"{metrics['cv_accuracy_mean']*100:.1f}%")
col3.metric("Diseases Covered", "20")

st.divider()

# ---------------------------------------------------------------------------
# Symptom selection
# ---------------------------------------------------------------------------
st.subheader("1. Select your symptoms")

all_symptoms = get_symptom_list()
display_names = {s: s.replace("_", " ").title() for s in all_symptoms}

cols = st.columns(3)
selected = []
for i, symptom in enumerate(all_symptoms):
    col = cols[i % 3]
    if col.checkbox(display_names[symptom], key=symptom):
        selected.append(symptom)

st.divider()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
st.subheader("2. Prediction")

if not selected:
    st.info("Select at least one symptom above to get a prediction.")
else:
    top_k = st.slider("Number of predictions to show", 1, 5, 3)
    results = predict(selected, top_k=top_k)

    df = pd.DataFrame(results, columns=["Disease", "Probability"])
    df["Probability (%)"] = (df["Probability"] * 100).round(1)

    fig = px.bar(
        df,
        x="Probability (%)",
        y="Disease",
        orientation="h",
        color="Probability (%)",
        color_continuous_scale="Reds",
        title="Predicted Likelihood by Disease",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3. Details & general guidance")
    for disease, prob in results:
        with st.expander(f"{disease} — {prob*100:.1f}% likelihood"):
            precautions = PRECAUTIONS.get(disease, [])
            if precautions:
                st.markdown("**General guidance** (not medical advice):")
                for p in precautions:
                    st.markdown(f"- {p}")
            else:
                st.markdown("No guidance available for this condition.")

st.divider()

# ---------------------------------------------------------------------------
# Model insight panel
# ---------------------------------------------------------------------------
with st.expander("📊 Model insight: most predictive symptoms overall"):
    top_symptoms = get_top_symptoms(10)
    imp_df = pd.DataFrame({
        "Symptom": [s.replace("_", " ").title() for s in top_symptoms.index],
        "Importance": top_symptoms.values,
    })
    fig2 = px.bar(imp_df, x="Importance", y="Symptom", orientation="h",
                  title="Top 10 Globally Important Symptoms (Feature Importance)")
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, use_container_width=True)
