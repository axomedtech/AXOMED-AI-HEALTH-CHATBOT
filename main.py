from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
import numpy as np  # type: ignore

# ─────────────────────────────────────────────
# 1. EXPANDED DATASET  (10 conditions, 50+ samples)
# ─────────────────────────────────────────────
sentences = [
    # FEVER
    "I have a fever",
    "high temperature",
    "I am feeling hot",
    "my body temperature is high",
    "I feel burning hot",
    "fever and chills",

    # COLD
    "I have a cold",
    "runny nose",
    "sneezing a lot",
    "blocked nose",
    "nasal congestion",
    "stuffy nose and sore throat",

    # HEADACHE
    "I have a headache",
    "my head hurts",
    "pain in my head",
    "head is throbbing",
    "pressure in my head",
    "constant head pain",

    # FLU
    "I have the flu",
    "body aches and fever",
    "muscle pain and tiredness",
    "I feel weak and feverish",
    "fatigue and body pain",
    "chills and sweating",

    # ALLERGY
    "I have allergies",
    "itchy eyes",
    "watery eyes and sneezing",
    "skin rash and itching",
    "hives on my skin",
    "allergic reaction",

    # COVID
    "loss of smell",
    "loss of taste",
    "I lost my sense of smell",
    "dry cough and fever",
    "shortness of breath and fever",
    "covid symptoms",

    # MIGRAINE
    "I have a migraine",
    "severe head pain on one side",
    "throbbing pain with nausea",
    "light sensitivity and headache",
    "I feel nauseous and my head hurts",
    "migraine attack",

    # FOOD POISONING
    "I feel nauseous after eating",
    "vomiting and diarrhea",
    "stomach cramps",
    "food poisoning symptoms",
    "upset stomach and vomiting",
    "diarrhea and stomach pain",

    # ASTHMA
    "difficulty breathing",
    "wheezing",
    "chest tightness",
    "shortness of breath",
    "I cannot breathe properly",
    "asthma attack",

    # ANXIETY
    "I feel anxious",
    "heart racing",
    "feeling nervous and restless",
    "panic attack",
    "I feel overwhelmed and stressed",
    "chest tightness and nervousness",
]

labels = (
    ["fever"] * 6
    + ["cold"] * 6
    + ["headache"] * 6
    + ["flu"] * 6
    + ["allergy"] * 6
    + ["covid"] * 6
    + ["migraine"] * 6
    + ["food_poisoning"] * 6
    + ["asthma"] * 6
    + ["anxiety"] * 6
)

# ─────────────────────────────────────────────
# 2. CONVERT TEXT → NUMBERS
# ─────────────────────────────────────────────
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# ─────────────────────────────────────────────
# 3. TRAIN MODEL
# ─────────────────────────────────────────────
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# ─────────────────────────────────────────────
# HELPER: draw a confidence bar
# ─────────────────────────────────────────────
BAR_WIDTH = 20

def confidence_bar(prob: float) -> str:
    filled = int(round(prob * BAR_WIDTH))
    return f"[{'█' * filled}{'░' * (BAR_WIDTH - filled)}] {prob * 100:.1f}%"

# ─────────────────────────────────────────────
# 4. CHAT LOOP
# ─────────────────────────────────────────────
print("=" * 50)
print("  🩺  Symptom Checker  —  Level 2")
print("=" * 50)
print("Describe your symptoms and I'll predict the condition.")
print("Type 'exit' to quit.\n")

while True:
    user = input("You: ").strip()

    if not user:
        continue
    if user.lower() == "exit":
        print("Take care! 👋")
        break

    X_test = vectorizer.transform([user])
    probs   = model.predict_proba(X_test)[0]          # probability for each class
    classes = model.classes_                           # class names in same order

    # Sort by probability descending
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    top_condition, top_prob = ranked[0]

    print()
    print(f"  🔍 Most likely: {top_condition.upper().replace('_', ' ')}  {confidence_bar(top_prob)}")
    print()
    print("  📊 Top 3 predictions:")
    for i, (condition, prob) in enumerate(ranked[:3], 1):
        label = condition.replace("_", " ").title()
        print(f"     {i}. {label:<18} {confidence_bar(prob)}")

    # Low-confidence warning
    if top_prob < 0.40:
        print()
        print("  ⚠️  Low confidence — try describing your symptoms in more detail.")

    print()