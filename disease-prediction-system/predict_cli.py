"""
predict_cli.py
---------------
Command-line interface to test predictions without launching Streamlit.

Usage:
    python predict_cli.py fever cough shortness_of_breath chest_pain
"""

import sys
from predictor import predict, get_symptom_list


def main():
    symptoms = sys.argv[1:]
    valid = set(get_symptom_list())

    if not symptoms:
        print("Usage: python predict_cli.py <symptom_1> <symptom_2> ...")
        print(f"\nAvailable symptoms:\n{', '.join(sorted(valid))}")
        return

    unknown = [s for s in symptoms if s not in valid]
    if unknown:
        print(f"Unknown symptom(s): {', '.join(unknown)}")
        print("Run with no arguments to see the full valid symptom list.")
        return

    results = predict(symptoms, top_k=3)
    print(f"\nInput symptoms: {', '.join(symptoms)}\n")
    print("Top predictions:")
    for disease, prob in results:
        print(f"  {disease:<25} {prob*100:.1f}%")


if __name__ == "__main__":
    main()
