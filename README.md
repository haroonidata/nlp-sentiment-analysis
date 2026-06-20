# NLP Sentiment Analysis

## Project Overview

This project builds a machine learning model to classify IMDb movie reviews as **Positive** or **Negative** using Natural Language Processing (NLP).

The project demonstrates a complete NLP pipeline from raw text through feature engineering, model training, evaluation, and interpretation.

---

## Dataset

IMDb Movie Reviews Dataset

* 50,000 movie reviews
* Balanced dataset
* 25,000 Positive reviews
* 25,000 Negative reviews

### Target Variable

| Label | Meaning  |
| ----- | -------- |
| 1     | Positive |
| 0     | Negative |

---

## NLP Pipeline

```text
Raw Review
↓
Text Cleaning
↓
TF-IDF Vectorisation
↓
Train/Test Split
↓
Logistic Regression
↓
Prediction
↓
Evaluation
```

---

## Text Preprocessing

The following preprocessing steps were applied:

* Convert text to lowercase
* Remove punctuation
* Remove numbers
* Remove special characters
* Remove stop words
* Convert reviews into TF-IDF features

---

## Model

### Feature Engineering

* TF-IDF Vectorizer
* Maximum Features: 10,000
* English Stop Words Removed

### Classification Model

* Logistic Regression
* Max Iterations: 1000

---

## Results

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.889 |
| Precision | 0.877 |
| Recall    | 0.907 |
| F1 Score  | 0.892 |

### Confusion Matrix

| Actual / Predicted | Negative | Positive |
| ------------------ | -------: | -------: |
| Negative           |     4322 |      639 |
| Positive           |      471 |     4568 |

---

## Most Positive Words

The model identified the following words as strong indicators of positive sentiment:

* great
* excellent
* best
* wonderful
* perfect
* amazing
* favorite
* brilliant
* loved
* enjoyed

---

## Most Negative Words

The model identified the following words as strong indicators of negative sentiment:

* worst
* waste
* awful
* bad
* boring
* poor
* terrible
* worse
* poorly
* dull

---

## Key NLP Concepts Covered

* Text Cleaning
* Stop Words
* TF-IDF
* Sparse Matrices
* Train/Test Split
* Logistic Regression
* Sigmoid Function
* Probabilities
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Feature Importance
* Model Interpretation

---

## Project Structure

```text
sentiment-analysis/
│
├── data/
├── models/
├── src/
│   ├── train.py
│   ├── preprocess.py
│   ├── evaluate.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

---

## Tech Stack

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression
* Joblib

---

## Future Improvements

* MLflow Experiment Tracking
* Naive Bayes Comparison
* Cross Validation
* N-Gram Features
* Word Embeddings
* Word2Vec
* Transformer Models
