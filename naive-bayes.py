import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class Naive_Bayes():
    def __init__(self, is_text_data=False, stop_words=None, min_df=1):
        """
        is_text_data: boolean indicating if the input is text data.
        stop_words: stop words for text vectorization (applicable for text data only).
        min_df: minimum document frequency for vectorization (applicable for text data only).
        """
        self.is_text_data = is_text_data
        self.stop_words = stop_words
        self.min_df = min_df
        self.vectorizer = None
        self.feature_likelihoods = {}
        self.class_priors = {}

    def _process_text(self, X):
        """
        vectorize text data
        """
        self.vectorizer = CountVectorizer(stop_words=self.stop_words, min_df=self.min_df)
        return self.vectorizer.fit_transform(X).toarray()

    def fit(self, X, y):
        if self.is_text_data:
            X = self._process_text(X)

        y = np.array(y)
        self.classes = np.unique(y)

        # calculate prior probabilities for each class
        total_samples = len(y)
        self.class_priors = {
            cls: np.sum(y == cls) / total_samples for cls in self.classes
        }

        # calculate likelihoods for each feature given each class
        for cls in self.classes:
            class_samples = X[y == cls]
            feature_sums = class_samples.sum(axis=0) + 1  # Add-one smoothing
            total_features = feature_sums.sum()
            self.feature_likelihoods[cls] = feature_sums / total_features

    def predict(self, X):
        if self.is_text_data:
            if not self.vectorizer:
                raise ValueError("Model not fitted for text data.")
            X = self.vectorizer.transform(X).toarray()

        predictions = []
        for sample in X:
            posteriors = {}
            for cls in self.classes:
                # compute posterior probability
                likelihood = self.feature_likelihoods[cls]
                power = likelihood ** sample
                prod_likelihood = power.prod()
                posteriors[cls] = prod_likelihood * self.class_priors[cls]

            # assign class with the highest posterior probability
            predictions.append(max(posteriors, key=posteriors.get))
        return predictions

