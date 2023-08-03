from typing import Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse as spr
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer

from topicwizard.pipeline import TopicPipeline, make_topic_pipeline


class SparseWithText(spr.csr_array):
    """Compressed Sparse Row sparse array with a text attribute,
    this way the textual content of the sparse array can be
    passed down in a pipeline."""

    def __init__(self, *args, texts: Optional[list[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if texts is None:
            self.texts = None
        else:
            self.texts = list(texts)


class LeakyCountVectorizer(CountVectorizer):
    """Leaky CountVectorizer class, that does essentially the exact same
    thing as scikit-learn's CountVectorizer, but returns a sparse
    array with the text attribute attached. (see SparseWithText)"""

    def fit_transform(self, raw_documents, y=None):
        raw_documents = list(raw_documents)
        res = super().fit_transform(raw_documents, y=y)
        return SparseWithText(res, texts=raw_documents)

    def transform(self, raw_documents):
        raw_documents = list(raw_documents)
        res = super().transform(raw_documents)
        return SparseWithText(res, texts=raw_documents)


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
        X = SparseWithText(X, texts=texts)
        return X

    def fit_transform(self, raw_documents: Iterable[str], y=None):
        raw_documents = list(raw_documents)
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

    def fit(self, X: SparseWithText, y=None):
        self.model.fit(X.texts)
        return self

    def transform(self, X: SparseWithText) -> np.array:
        dist, _ = self.model.approximate_distribution(X.texts, padding=True)
        return np.array(dist)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


def bertopic_pipeline(model) -> TopicPipeline:
    """Creates sklearn compatible wrapper for a BERTopic topic pipeline.

    Parameters
    ----------
    model: BERTopic
        Any BERTopic model.

    Returns
    -------
    pipeline: TopicPipeline
        Sklearn compatible topic pipeline wrapping the BERTopic topic model.
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
    pipeline = make_topic_pipeline(vectorizer, topic_model)
    pipeline.topic_names = topic_names
    return pipeline
