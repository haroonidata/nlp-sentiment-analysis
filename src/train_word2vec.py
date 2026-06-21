import pandas as pd
import numpy as np
from preprocess import clean_text
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from evaluate import evaluate_model
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn
import joblib

mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("sentiment-analysis")
df = pd.read_csv("../data/imdb_reviews.csv")


df["clean_review"] = df["review"].apply(clean_text)

df["tokens"] = df["clean_review"].str.split()

print(df[["clean_review", "tokens"]].head())

sentences = df["tokens"].tolist()

word2vec_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

print(word2vec_model.wv["great"])
print(word2vec_model.wv.most_similar("great"))
def get_review_vector(tokens, model):
    vectors = []

    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


X = np.array([
    get_review_vector(tokens, word2vec_model)
    for tokens in df["tokens"]
])

y = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

print(X.shape)
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

metrics = evaluate_model(y_test, predictions)
print(metrics)

train_predictions = model.predict(X_train)

print("Train F1:", f1_score(y_train, train_predictions))
print("Test F1:", f1_score(y_test, predictions))

with mlflow.start_run(run_name="word2vec_avg_logreg"):
    mlflow.log_param("embedding", "word2vec")
    mlflow.log_param("vector_size", 100)
    mlflow.log_param("window", 5)
    mlflow.log_param("min_count", 2)
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_param("review_vector_method", "average")

    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("precision", metrics["precision"])
    mlflow.log_metric("recall", metrics["recall"])
    mlflow.log_metric("f1", metrics["f1"])
    mlflow.log_metric("train_f1", f1_score(y_train, train_predictions))
    mlflow.log_metric("test_f1", f1_score(y_test, predictions))


word2vec_model.save("../models/word2vec.model")

joblib.dump(
    model,
    "../models/word2vec_logreg_model.pkl"
)