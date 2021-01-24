from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from collections import Counter
import numpy as np
from hazm import *
import re

np.random.seed(42)

stemmer = Stemmer()
class CommentToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_punctuation=True, stemming=True):
        self.remove_punctuation = remove_punctuation
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for comment in X:
            if self.remove_punctuation:
                text = re.sub('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890^\w\s#@/:%.,_-', '', comment, flags=re.M)
                word_counts = Counter(word_tokenize(text))
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)
