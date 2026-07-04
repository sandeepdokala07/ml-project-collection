"""
generate_dataset.py
--------------------
Builds a labeled symptom -> disease dataset from a curated clinical knowledge
base (disease -> characteristic symptom set), then synthesizes realistic
training samples by sampling subsets of each disease's symptoms plus small
amounts of noise (co-occurring/overlapping symptoms), mimicking how real
patient-reported symptom data looks.

NOTE: This is a compact educational knowledge base (20 common diseases,
50 symptoms) meant to demonstrate the full ML pipeline end-to-end. For a
production or research-grade system, replace `DISEASE_SYMPTOM_MAP` with a
vetted clinical dataset (e.g. a licensed symptom-disease corpus reviewed by
a medical professional) rather than expanding this by hand.
"""

import random
import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------------
# Full symptom vocabulary (feature columns)
# ---------------------------------------------------------------------------
SYMPTOMS = [
    "fever", "cough", "fatigue", "headache", "sore_throat", "runny_nose",
    "body_ache", "chills", "shortness_of_breath", "loss_of_taste",
    "loss_of_smell", "nausea", "vomiting", "diarrhea", "abdominal_pain",
    "joint_pain", "muscle_pain", "rash", "high_blood_sugar",
    "frequent_urination", "excessive_thirst", "blurred_vision",
    "high_blood_pressure", "dizziness", "chest_pain", "wheezing",
    "night_sweats", "weight_loss", "weakness", "pale_skin", "sneezing",
    "itchy_eyes", "mucus_production", "sputum_production",
    "blood_in_sputum", "low_appetite", "jaundice", "abdominal_cramps",
    "burning_urination", "cloudy_urine", "swelling", "stiffness",
    "mood_swings", "hair_loss", "weight_gain", "cold_intolerance",
    "heat_intolerance", "tremors", "palpitations", "sensitivity_to_light",
    "stiff_neck",
]

# ---------------------------------------------------------------------------
# Disease -> characteristic symptoms (core = usually present, minor = sometimes)
# ---------------------------------------------------------------------------
DISEASE_SYMPTOM_MAP = {
    "Common Cold": {
        "core": ["runny_nose", "sneezing", "sore_throat", "cough"],
        "minor": ["headache", "fatigue", "low_appetite"],
    },
    "Influenza (Flu)": {
        "core": ["fever", "body_ache", "chills", "fatigue", "headache"],
        "minor": ["cough", "sore_throat", "runny_nose"],
    },
    "COVID-19": {
        "core": ["fever", "cough", "fatigue", "loss_of_taste", "loss_of_smell"],
        "minor": ["shortness_of_breath", "body_ache", "sore_throat"],
    },
    "Migraine": {
        "core": ["headache", "sensitivity_to_light", "nausea"],
        "minor": ["dizziness", "fatigue"],
    },
    "Diabetes": {
        "core": ["high_blood_sugar", "frequent_urination", "excessive_thirst", "blurred_vision"],
        "minor": ["fatigue", "weight_loss"],
    },
    "Hypertension": {
        "core": ["high_blood_pressure", "headache", "dizziness"],
        "minor": ["chest_pain", "fatigue"],
    },
    "Asthma": {
        "core": ["wheezing", "shortness_of_breath", "cough"],
        "minor": ["chest_pain", "fatigue"],
    },
    "Pneumonia": {
        "core": ["fever", "cough", "shortness_of_breath", "chest_pain", "sputum_production"],
        "minor": ["fatigue", "chills"],
    },
    "Bronchitis": {
        "core": ["cough", "mucus_production", "fatigue", "wheezing"],
        "minor": ["sore_throat", "low_appetite"],
    },
    "Malaria": {
        "core": ["fever", "chills", "night_sweats", "headache", "body_ache"],
        "minor": ["nausea", "fatigue"],
    },
    "Dengue": {
        "core": ["fever", "body_ache", "rash", "joint_pain"],
        "minor": ["headache", "nausea", "fatigue"],
    },
    "Typhoid": {
        "core": ["fever", "abdominal_pain", "low_appetite", "weakness"],
        "minor": ["headache", "diarrhea"],
    },
    "Chickenpox": {
        "core": ["rash", "fever", "itchy_eyes", "fatigue"],
        "minor": ["headache", "low_appetite"],
    },
    "Tuberculosis": {
        "core": ["cough", "blood_in_sputum", "weight_loss", "night_sweats"],
        "minor": ["fatigue", "fever"],
    },
    "Anemia": {
        "core": ["fatigue", "pale_skin", "weakness", "dizziness"],
        "minor": ["headache", "palpitations"],
    },
    "Gastroenteritis": {
        "core": ["vomiting", "diarrhea", "abdominal_cramps", "nausea"],
        "minor": ["fever", "weakness"],
    },
    "Urinary Tract Infection": {
        "core": ["burning_urination", "frequent_urination", "cloudy_urine", "abdominal_pain"],
        "minor": ["fever", "fatigue"],
    },
    "Hypothyroidism": {
        "core": ["weight_gain", "cold_intolerance", "fatigue", "hair_loss"],
        "minor": ["mood_swings", "muscle_pain"],
    },
    "Hyperthyroidism": {
        "core": ["weight_loss", "heat_intolerance", "tremors", "palpitations"],
        "minor": ["mood_swings", "fatigue"],
    },
    "Arthritis": {
        "core": ["joint_pain", "stiffness", "swelling"],
        "minor": ["fatigue", "muscle_pain"],
    },
}

# Simple, non-prescriptive general guidance (NOT medical advice) shown in the UI.
PRECAUTIONS = {
    "Common Cold": ["Rest and stay hydrated", "Warm fluids for sore throat", "See a doctor if symptoms persist beyond 10 days"],
    "Influenza (Flu)": ["Rest and isolate to avoid spreading", "Stay hydrated", "Consult a doctor if fever is high or persistent"],
    "COVID-19": ["Isolate per local health guidance", "Monitor oxygen levels if available", "Seek medical care if breathing difficulty occurs"],
    "Migraine": ["Rest in a dark, quiet room", "Stay hydrated", "Consult a doctor for recurrent migraines"],
    "Diabetes": ["Consult a doctor for blood sugar testing", "Monitor diet and hydration", "Seek care for vision changes"],
    "Hypertension": ["Monitor blood pressure regularly", "Reduce salt intake", "Consult a doctor for persistent high readings"],
    "Asthma": ["Avoid known triggers", "Keep prescribed inhalers accessible", "Seek urgent care for severe breathlessness"],
    "Pneumonia": ["Seek medical attention promptly", "Rest and stay hydrated", "Monitor breathing difficulty closely"],
    "Bronchitis": ["Rest and stay hydrated", "Avoid smoke/irritants", "See a doctor if cough persists over 3 weeks"],
    "Malaria": ["Seek medical testing and treatment promptly", "Use mosquito protection", "Stay hydrated"],
    "Dengue": ["Seek medical care promptly", "Stay hydrated", "Avoid aspirin/ibuprofen without medical advice"],
    "Typhoid": ["Seek medical testing and antibiotics if confirmed", "Maintain hydration", "Practice food/water hygiene"],
    "Chickenpox": ["Avoid scratching rash", "Isolate to prevent spread", "Consult a doctor, especially for adults"],
    "Tuberculosis": ["Seek medical testing promptly", "Complete full prescribed treatment course", "Avoid close contact until cleared"],
    "Anemia": ["Consult a doctor for blood tests", "Iron-rich diet as advised by a professional", "Monitor fatigue levels"],
    "Gastroenteritis": ["Stay hydrated with oral rehydration", "Rest the gut with bland food", "Seek care if symptoms are severe or prolonged"],
    "Urinary Tract Infection": ["Stay hydrated", "Seek medical evaluation for antibiotics", "Avoid holding urine for long periods"],
    "Hypothyroidism": ["Consult a doctor for thyroid function tests", "Discuss medication options with a physician", "Monitor weight and energy changes"],
    "Hyperthyroidism": ["Consult a doctor for thyroid function tests", "Monitor heart rate", "Discuss treatment options with a physician"],
    "Arthritis": ["Gentle regular movement as tolerated", "Consult a doctor for pain management", "Apply heat/cold as advised"],
}


def generate_samples(disease: str, symptom_info: dict, n_samples: int = 60) -> list:
    """Synthesize patient-like symptom rows for one disease."""
    core = symptom_info["core"]
    minor = symptom_info["minor"]
    other_symptoms = [s for s in SYMPTOMS if s not in core and s not in minor]

    rows = []
    for _ in range(n_samples):
        row = {s: 0 for s in SYMPTOMS}

        # Core symptoms: usually present (85-100% chance each)
        for s in core:
            if random.random() < 0.9:
                row[s] = 1

        # Minor symptoms: sometimes present (40-60% chance each)
        for s in minor:
            if random.random() < 0.5:
                row[s] = 1

        # Noise: rare unrelated symptom (simulates comorbidity / reporting noise)
        for s in random.sample(other_symptoms, k=random.randint(0, 2)):
            row[s] = 1

        row["disease"] = disease
        rows.append(row)
    return rows


def main():
    all_rows = []
    for disease, info in DISEASE_SYMPTOM_MAP.items():
        all_rows.extend(generate_samples(disease, info, n_samples=60))

    df = pd.DataFrame(all_rows)
    # Reorder columns: symptoms first, disease last
    df = df[SYMPTOMS + ["disease"]]
    df.to_csv("data/symptom_disease_dataset.csv", index=False)
    print(f"Generated dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Diseases: {df['disease'].nunique()} | Symptoms: {len(SYMPTOMS)}")


if __name__ == "__main__":
    main()
