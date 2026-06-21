import joblib

from preprocess import clean_text

model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")


def predict_sentiment(review):
    clean_review = clean_text(review)

    X = vectorizer.transform([clean_review])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    if prediction == 1:
        label = "Positive"
        confidence = probability[1]
    else:
        label = "Negative"
        confidence = probability[0]

    return label, confidence


review = "This movie was absolutely amazing and I loved it"

label, confidence = predict_sentiment(review)

print("Review:", review)
print("Prediction:", label)
print(f"Confidence: {confidence:.4f}")