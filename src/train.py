import pandas as pd


from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from evaluate import evaluate_model
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import joblib

mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("sentiment-analysis")
df = pd.read_csv("../data/imdb_reviews.csv")

df["clean_review"] = df["review"].apply(clean_text)

print(df[["review", "clean_review"]].head())

print(df.shape)

print(df["sentiment"].value_counts())
df["label"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

vectorizer = TfidfVectorizer(
    stop_words=None,
    max_features=10000,
    ngram_range=(1,2)

)

X = vectorizer.fit_transform(df["clean_review"])
y = df["label"]

print(df.head())
print(df["label"].value_counts())
print(X.shape)
print(vectorizer.get_feature_names_out()[:20])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(
    max_iter=1000
)

# model = MultinomialNB()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
metrics = evaluate_model(y_test, predictions)
print(metrics)
# feature_names = vectorizer.get_feature_names_out()

# weights_df = pd.DataFrame({
#     "word": feature_names,
#     "weight": model.coef_[0]
# })

# print("\nTop Positive Words:")

# print(
#     weights_df
#     .sort_values("weight", ascending=False)
#     .head(20)
# )

# print("\nTop Negative Words:")

# print(
#     weights_df
#     .sort_values("weight")
#     .head(20)
# )


train_predictions = model.predict(X_train)

print(
    "Train F1:",
    f1_score(y_train, train_predictions)
)

print(
    "Test F1:",
    f1_score(y_test, predictions)
)


joblib.dump(model, "../models/sentiment_model.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
print(vectorizer)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("Cross Validation F1 Scores:")
print(cv_scores)

print(
    "Average CV F1:",
    cv_scores.mean()
)

results_df = pd.DataFrame({
    "review": df.loc[y_test.index, "review"],
    "actual": y_test,
    "predicted": predictions
})

errors = results_df[
    results_df["actual"] != results_df["predicted"]
]

print(errors.head(20))
train_f1 = f1_score(y_train, train_predictions)
test_f1 = f1_score(y_test, predictions)
print("Train F1:", train_f1)
print("Test F1:", test_f1)
with mlflow.start_run(run_name="logistic_regression_bigrams"):
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_param("stop_words",None)
    mlflow.log_param("max_features", 10000)

    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("precision", metrics["precision"])
    mlflow.log_metric("recall", metrics["recall"])
    mlflow.log_metric("f1", metrics["f1"])

    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_metric("train_f1", train_f1)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("cv_mean_f1", cv_scores.mean())