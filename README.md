"""Run known symptom sets through the model and report top-3 results.

Usage (from project root):  python -m src.test_examples
Put this file in src/.
"""
from src.predict import load_artifacts, predict_top_k

CASES = [
    ("Fungal infection", ["itching", "skin_rash", "nodal_skin_eruptions"]),
    ("Common Cold", ["continuous_sneezing", "chills", "cough", "runny_nose",
                     "congestion", "throat_irritation"]),
    ("Malaria", ["chills", "vomiting", "high_fever", "sweating", "headache",
                 "nausea", "diarrhoea", "muscle_pain"]),
    ("Dengue", ["skin_rash", "joint_pain", "high_fever", "pain_behind_the_eyes",
                "red_spots_over_body", "back_pain"]),
    ("Jaundice", ["itching", "vomiting", "yellowish_skin", "dark_urine",
                  "abdominal_pain", "weight_loss"]),
    ("Diabetes ", ["fatigue", "weight_loss", "restlessness", "irregular_sugar_level",
                   "excessive_hunger", "polyuria"]),
    ("Hypertension", ["headache", "chest_pain", "dizziness", "loss_of_balance",
                      "lack_of_concentration"]),
]


def main():
    _, columns = load_artifacts()
    valid = set(columns)
    passed = 0

    for expected, symptoms in CASES:
        missing = [s for s in symptoms if s not in valid]
        if missing:
            print(f"[SKIP] {expected}: unknown columns {missing}\n")
            continue
        top3 = predict_top_k(symptoms, k=3)
        ok = top3[0][0].strip().lower() == expected.strip().lower()
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] expected: {expected.strip()}")
        for name, conf in top3:
            print(f"        {name}: {conf}%")
        print()

    # Edge case: a single vague symptom should give low confidence
    name, conf = predict_top_k(["headache"], k=1)[0]
    print(f"[INFO] only 'headache' -> {name} at {conf}% (expected to be low)\n")

    print(f"{passed}/{len(CASES)} cases passed")


if __name__ == "__main__":
    main()