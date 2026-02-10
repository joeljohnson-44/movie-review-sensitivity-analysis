# Import libraries
from abc import ABC, abstractmethod
import numpy as np
import torch

import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

nltk.download('vader_lexicon')


class SentimentModel(ABC):
    """
    Abstract base class defining a common interface for all sentiment models.
    All sentiment models (rule-based, ML, DL) must inherit from this class.
    """

    @abstractmethod
    def train(self, X, y):
        """
        Train the model on input text and labels.

        Parameters
        ----------
        X : iterable
            Collection of text samples
        y : iterable
            Corresponding sentiment labels
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict sentiment labels for given input texts.

        Parameters
        ----------
        X : iterable
            Collection of text samples

        Returns
        -------
        array-like
            Predicted sentiment labels
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Save the trained model to disk.

        Parameters
        ----------
        path : str
            File path to save the model
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        path : str
            File path from which to load the model
        """
        pass

    def evaluate(self, X, y_true):
        """
        Evaluate the model on test data.

        Parameters
        ----------
        X : iterable
            Test text samples
        y_true : iterable
            True labels

        Returns
        -------
        dict
            Dictionary containing accuracy, classification report, and confusion matrix
        """
        y_pred = self.predict(X)

        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred)
        }

        return results

    def print_evaluation(self, X, y_true):
        """
        Print evaluation metrics in a readable format.

        Parameters
        ----------
        X : iterable
            Test text samples
        y_true : iterable
            True labels
        """
        results = self.evaluate(X, y_true)

        print("Accuracy:", results["accuracy"])
        print("F1-Score:", results["f1_score"])
        print("\nClassification Report:\n")
        print(results["classification_report"])
        print("\nConfusion Matrix:\n")
        print(results["confusion_matrix"])


class NLTKSentimentModel(SentimentModel):
    """
    Rule-based sentiment classifier using NLTK VADER.
    Does not require training.
    """

    def __init__(self):
        """Initialize the VADER sentiment analyzer."""
        self.sia = SentimentIntensityAnalyzer()

    def train(self, X, y):
        """No training required for VADER-based model."""
        print("VADER model does not require training.")

    def vader_predict_single(self, text):
        """
        Predict sentiment for a single text input.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        int
            1 for positive sentiment, 0 for negative
        """
        score = self.sia.polarity_scores(text)['compound']
        return 1 if score >= 0 else 0

    def predict(self, X):
        """
        Predict sentiment for multiple texts.

        Parameters
        ----------
        X : pandas Series or list
            Input texts

        Returns
        -------
        pandas Series
            Predicted labels
        """
        return X.apply(self.vader_predict_single)

    def save(self, path):
        """Save placeholder for API consistency."""
        joblib.dump("vader", path)

    def load(self, path):
        """Reload the VADER analyzer."""
        self.sia = SentimentIntensityAnalyzer()


class TfidfLogisticSentimentModel(SentimentModel):
    """
    TF-IDF + Logistic Regression based sentiment classifier.
    """

    def __init__(self, max_features=30000, ngram_range=(1, 2)):
        """
        Initialize TF-IDF Logistic Regression model.

        Parameters
        ----------
        max_features : int
            Maximum number of features in TF-IDF
        ngram_range : tuple
            Range of n-grams to use
        """
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english'
            )),
            ('clf', LogisticRegression(max_iter=1000))
        ])

    def train(self, X, y):
        """Train the pipeline model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict labels for input texts."""
        return self.model.predict(X)

    def save(self, path):
        """Save the trained pipeline."""
        joblib.dump(self.model, path)

    def load(self, path):
        """Load saved pipeline."""
        self.model = joblib.load(path)


class TfidfRandomForestSentimentModel(SentimentModel):
    """
    TF-IDF + Random Forest based sentiment classifier.
    """

    def __init__(self, max_features=30000, ngram_range=(1, 2), n_estimators=200):
        """Initialize Random Forest pipeline."""
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english'
            )),
            ('clf', RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            ))
        ])

    def train(self, X, y):
        """Train the model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict sentiment labels."""
        return self.model.predict(X)

    def save(self, path):
        """Save model pipeline."""
        joblib.dump(self.model, path)

    def load(self, path):
        """Load saved model pipeline."""
        self.model = joblib.load(path)


class TfidfXGBoostSentimentModel(SentimentModel):
    """
    TF-IDF + XGBoost based sentiment classifier.
    """

    def __init__(self, max_features=30000, ngram_range=(1, 2)):
        """Initialize XGBoost pipeline."""
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english'
            )),
            ('clf', XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                eval_metric='logloss',
                n_jobs=-1
            ))
        ])

    def train(self, X, y):
        """Train XGBoost model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict sentiment labels."""
        return self.model.predict(X)

    def save(self, path):
        """Save trained model."""
        joblib.dump(self.model, path)

    def load(self, path):
        """Load saved model."""
        self.model = joblib.load(path)


class TransformerSentimentModel(SentimentModel):
    """
    Transformer-based sentiment classifier using HuggingFace models.
    This version trains only on training data without validation set.
    """

    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        """Initialize tokenizer and transformer model with proper device handling."""

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # Move model to appropriate device
        self.model.to(self.device)

        self.trainer = None
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _prepare_dataset(self, X, y):
        """
        Convert raw text and labels to tokenized dataset.

        Parameters
        ----------
        X : iterable
            Input text samples
        y : iterable
            Corresponding labels

        Returns
        -------
        Dataset
            Tokenized HuggingFace dataset
        """
        dataset = Dataset.from_dict({
            "text": list(X),
            "label": list(y)
        })

        return dataset.map(
            lambda e: self.tokenizer(e["text"], truncation=True),
            batched=True
        )

    def train(self, X, y, epochs=2):
        """
        Train transformer model using only training data.

        Parameters
        ----------
        X : iterable
            Training text samples
        y : iterable
            Training labels
        epochs : int
            Number of training epochs
        """
        train_dataset = self._prepare_dataset(X, y)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            logging_steps=50
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=self.data_collator
        )

        self.trainer.train()

    def predict(self, X):
        """
        Predict sentiment labels for input texts.

        Parameters
        ----------
        X : iterable
            Text samples to classify

        Returns
        -------
        array-like
            Predicted labels
        """
        texts = list(X)

        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, axis=1)

        # Return results on CPU
        return predictions.cpu().numpy()

    def save(self, path):
        """
        Save transformer model and tokenizer.

        Parameters
        ----------
        path : str
            Directory path to save model
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        """
        Load transformer model and tokenizer.

        Parameters
        ----------
        path : str
            Directory path containing saved model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)

        # Ensure loaded model is on correct device
        self.model.to(self.device)
