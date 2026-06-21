import numpy as np
import joblib

from gensim.models import Word2Vec
from preprocess import clean_text

word2vec_model = Word2Vec.load("../models/word2vec.model")
model = joblib.load("../models/word2vec_logreg_model.pkl")


def get_review_vector(tokens, word2vec_model):
    vectors = []

    for word in tokens:
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])

    if len(vectors) == 0:
        return np.zeros(word2vec_model.vector_size)

    return np.mean(vectors, axis=0)


def predict_sentiment(review):
    clean_review = clean_text(review)
    tokens = clean_review.split()

    review_vector = get_review_vector(tokens, word2vec_model)

    X = review_vector.reshape(1, -1)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    if prediction == 1:
        label = "Positive"
        confidence = probability[1]
    else:
        label = "Negative"
        confidence = probability[0]

    return label, confidence


review = input("Review: ")

label, confidence = predict_sentiment(review)

print("Prediction:", label)
print(f"Confidence: {confidence:.4f}")