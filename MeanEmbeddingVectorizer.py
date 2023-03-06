import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = self.word2vec.vector_size

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec.get_vector(w) for w in words if w in self.word2vec.vocab] or [np.zeros(self.dim)],
                    axis=0)
            for words in X
        ])
