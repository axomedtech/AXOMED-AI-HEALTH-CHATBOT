from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore

# 1. DATA (very small dataset)
sentences = [
    "I have fever",
    "high temperature",
    "I am feeling hot",
    "I have cold",
    "runny nose",
    "sneezing",
    "headache",
    "my head hurts"
]

labels = [
    "fever",
    "fever",
    "fever",
    "cold",
    "cold",
    "cold",
    "headache",
    "headache"
]

# 2. CONVERT TEXT → NUMBERS
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# 3. TRAIN MODEL
model = LogisticRegression()
model.fit(X, labels)

print("✅ AI trained!")

# 4. TEST AI
while True:
    user = input("You: ")
    
    if user == "exit":
        break

    X_test = vectorizer.transform([user])
    prediction = model.predict(X_test)

    print("AI:", prediction[0])