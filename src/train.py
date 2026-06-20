import pandas as pd


from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from evaluate import evaluate_model

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
    stop_words="english",
    max_features=10000
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

model.fit(X_train, y_train)

predictions = model.predict(X_test)
metrics = evaluate_model(y_test, predictions)

feature_names = vectorizer.get_feature_names_out()

weights_df = pd.DataFrame({
    "word": feature_names,
    "weight": model.coef_[0]
})

print("\nTop Positive Words:")

print(
    weights_df
    .sort_values("weight", ascending=False)
    .head(20)
)

print("\nTop Negative Words:")

print(
    weights_df
    .sort_values("weight")
    .head(20)
)