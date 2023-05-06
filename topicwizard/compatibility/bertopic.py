from typing import Callable, Iterable

import numpy as np
from bertopic import BERTopic
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline


class _BERTopicVectorizer(BaseEstimator):
    def __init__(self, topic_model: BERTopic, set_texts: Callable):
        self.vectorizer = topic_model.vectorizer_model
        self.set_texts = set_texts
        self.vocabulary_ = self.vectorizer.vocabulary_
        self.stop_words_ = self.vectorizer.stop_words_
        self.fixed_vocabulary_ = self.vectorizer.fixed_vocabulary_

    def fit(self, raw_documents: Iterable[str], y=None):
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, raw_documents: Iterable[str]):
        self.set_texts(raw_documents)
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, raw_documents: Iterable[str], y=None):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()


class _BERTopicModel(BaseEstimator):
    def __init__(self, topic_model: BERTopic, get_texts: Callable):
        self.model = topic_model
        self.get_texts = get_texts

    def fit(self, X, y=None):
        documents = self.get_texts()
        self.model.fit(documents)
        return self

    def transform(self, X) -> np.array:
        documents = self.get_texts()
        return self.model.transform(documents)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class TextContainer:
    def __init__(self):
        self.texts = []

    def set_texts(self, texts: Iterable[str]):
        self.texts = texts

    def get_texts(self) -> Iterable[str]:
        return self.texts


def bertopic_pipeline(topic_model: BERTopic) -> Pipeline:
    """Creates sklearn compatible wrapper for a BERTopic topic pipeline.

    Parameters
    ----------
    topic_model: BERTopic
        Any BERTopic model.

    Returns
    -------
    Pipeline
        Sklearn pipeline wrapping the BERTopic topic model.
    """
    # I create a closure, so the raw documents can be
    # Passed through the pipeline without the intermediary step
    # This is necessary, because BERTopic doesn't produce
    # Embeddings from the BOW representation, but embeds texts
    # with transformers.
    texts = TextContainer()
    vectorizer = _BERTopicVectorizer(topic_model=topic_model, texts.set_texts=set_texts)
    model = _BERTopicModel(topic_model=topic_model, get_texts=texts.get_texts)
    return make_pipeline(vectorizer, model)
