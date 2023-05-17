from typing import Iterable, List, Tuple

import numpy as np
import scipy.sparse as spr
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline


class _SparseWithText(spr.csr_array):
    def __init__(self, *args, texts: Iterable[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if texts is None:
            self.texts = None
        else:
            self.texts = list(texts)


class _BERTopicVectorizer(BaseEstimator):
    def __init__(self, topic_model):
        self.topic_model = topic_model
        self.vectorizer = topic_model.vectorizer_model
        self.vocabulary_ = self.vectorizer.vocabulary_
        self.stop_words_ = self.vectorizer.stop_words_
        self.fixed_vocabulary_ = self.vectorizer.fixed_vocabulary_

    def fit(self, raw_documents: Iterable[str], y=None):
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, raw_documents: Iterable[str]):
        texts = list(raw_documents)
        X = self.vectorizer.transform(texts)
        X = _SparseWithText(X, texts=texts)
        return X

    def fit_transform(self, raw_documents: Iterable[str], y=None):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()


class _BERTopicModel(BaseEstimator):
    def __init__(self, topic_model):
        self.topic_model = topic_model
        self.model = topic_model

    @property
    def _has_outliers(self):
        return -1 in self.model.topic_labels_

    @property
    def components_(self):
        ctfidf = self.model.c_tf_idf_
        if self._has_outliers:
            ctfidf = ctfidf[1:, :]
        return ctfidf.toarray()

    def fit(self, X: _SparseWithText, y=None):
        self.model.fit(X.texts)
        return self

    def transform(self, X: _SparseWithText) -> np.array:
        dist, _ = self.model.approximate_distribution(X.texts, padding=True)
        return np.array(dist)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


def bertopic_pipeline(model) -> Tuple[Pipeline, List[str]]:
    """Creates sklearn compatible wrapper for a BERTopic topic pipeline.

    Parameters
    ----------
    model: BERTopic
        Any BERTopic model.

    Returns
    -------
    pipeline: Pipeline
        Sklearn pipeline wrapping the BERTopic topic model.
    topic_names: list of str
        Names of topics assigned in the BERTopic model.
    """
    vectorizer = _BERTopicVectorizer(topic_model=model)
    topic_model = _BERTopicModel(topic_model=model)
    n_components, _ = topic_model.components_.shape
    topic_names = []
    if model.custom_labels_ is not None:
        topic_labels = model.custom_labels_
    else:
        topic_labels = model.topic_labels_
    for i_topic in range(n_components):
        topic_names.append(topic_labels[i_topic])
    return make_pipeline(vectorizer, topic_model), topic_names
