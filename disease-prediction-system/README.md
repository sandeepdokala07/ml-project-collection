# ML-Based Disease Prediction System

A machine learning system that predicts the most likely disease(s) from a set
of reported symptoms, using a **Random Forest classifier**, with a
**Streamlit** dashboard for interactive use.

This is a tabular/clinical-ML complement to a computer-vision-based medical
project (e.g. an image-based diagnosis system) — good for showing breadth
across ML problem types: **structured/tabular classification** here vs.
**deep learning / image classification** elsewhere in a portfolio.

## How it works

```
generate_dataset.py   →  builds a symptom-disease dataset from a curated
                          clinical knowledge base (20 diseases x 51 symptoms)
                                      │
                                      ▼
train_model.py         →  trains a Random Forest classifier, evaluates with
                           train/test split + 5-fold cross-validation,
                           saves model + feature importances to disk
                                      │
                                      ▼
predictor.py            →  loads the saved model, turns a symptom checklist
                            into ranked disease probabilities
                                      │
                                      ▼
app.py (Streamlit)      →  interactive symptom checklist, probability chart,
                            general precaution guidance, model insight panel
```

## Results (on the included dataset)

- **Test accuracy:** ~97.5%
- **5-fold cross-validation accuracy:** ~97.8% (± 0.5%)
- 20 diseases covered, 51 tracked symptoms, 1,200 training samples

Full per-class precision/recall is printed by `train_model.py`.

## Project structure

```
disease-prediction-system/
├── app.py                  # Streamlit dashboard (main entry point)
├── predict_cli.py          # CLI alternative — no UI needed
├── generate_dataset.py     # Builds the symptom-disease dataset + precautions dict
├── train_model.py          # Trains, evaluates, and saves the Random Forest model
├── predictor.py            # Loads saved model, exposes predict()
├── requirements.txt
├── data/
│   └── symptom_disease_dataset.csv   (generated)
├── models/
│   └── disease_model.pkl             (generated)
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Regenerate data & retrain (already done once, artifacts are included)

```bash
python generate_dataset.py   # writes data/symptom_disease_dataset.csv
python train_model.py        # writes models/disease_model.pkl, prints metrics
```

## Run the dashboard

```bash
streamlit run app.py
```

Check off symptoms in the UI to see ranked disease predictions with
probability scores and general (non-prescriptive) guidance.

## Run from the command line

```bash
python predict_cli.py fever chills night_sweats headache body_ache
# -> Malaria 99.0%, Influenza (Flu) 0.5%, Typhoid 0.5%
```

Run with no arguments to see the full list of valid symptom names.

## Important note on the dataset

`generate_dataset.py` synthesizes training samples from a **hand-curated
knowledge base** (`DISEASE_SYMPTOM_MAP`) of characteristic symptoms per
disease — it is built for demonstrating the full ML pipeline (data → model →
explainable predictions → UI), not for real diagnostic use. The very high
accuracy reflects the fact that the synthetic data is generated from clean
symptom rules; real-world clinical symptom data is noisier and would need:
- A vetted, clinically-reviewed dataset (e.g. hospital records or a licensed
  symptom-disease corpus) instead of synthetic rules.
- Handling of symptom co-occurrence across multiple simultaneous conditions.
- Calibration and validation by medical professionals before any real use.

The app includes an explicit on-screen disclaimer for this reason.

## Extending this project

- Swap Random Forest for a gradient-boosted model (XGBoost/LightGBM) and
  compare performance.
- Add SHAP values for per-prediction explainability (which symptoms drove
  *this* specific prediction, not just global importance).
- Expand `DISEASE_SYMPTOM_MAP` with more diseases/symptoms or replace it with
  a real dataset (e.g. the public Kaggle "Disease Prediction Using Machine
  Learning" symptom dataset) for a stronger resume story on data sourcing.
- Add severity scoring or triage logic (e.g. flag combinations that suggest
  urgent care).
