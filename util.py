import gensim
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB


def sentence_to_vector(sentence, word_embedding):
    vector = []
    for word in sentence.split():
        if word in word_embedding.wv:
            vector.append(word_embedding.wv[word])
    return np.mean(vector, axis=0)


def train_bernoulli(word_embedding_path, X_train, X_test, y_train, y_test):
    word_embedding = gensim.models.KeyedVectors.load(word_embedding_path)

    train_vectors = [sentence_to_vector(sentence, word_embedding) for sentence in X_train]
    test_vectors = [sentence_to_vector(sentence, word_embedding) for sentence in X_test]

    nb_model = BernoulliNB()
    nb_model.fit(train_vectors, y_train)

    y_pred = nb_model.predict(test_vectors)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    return nb_model
