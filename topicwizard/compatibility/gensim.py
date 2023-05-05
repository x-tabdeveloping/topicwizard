import warnings
from typing import Sequence

import numpy as np
import scipy.sparse as spr
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel, LdaMulticore, LsiModel, Nmf
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline


class DictionaryVectorizer(BaseEstimator):
    """Wrapper for Gensim's dictionary object as an sklearn compatible vectorizer object.

    Parameters
    ----------
    dictionary: gensim.corpora.dictionary.Dictionary
        Gensim's dictionary object. Please only pass already fitted
        dictionary objects.

    Attributes
    ----------
    index_to_key: array of shape (n_features)
        Mapping of feature indices to Gensim dictionary IDs.
    key_to_index: dict of int to int
        Mapping of Gensim Dictionary IDs to feature indices.
    vocabulary_: dict of str to int
        Mapping of terms to feature indices.
    """

    def __init__(self, dictionary: Dictionary):
        self.dictionary = dictionary
        self.index_to_key = np.array(dictionary.keys())
        self.key_to_index = {key: index for index, key in enumerate(self.key_to_index)}
        self.feature_names_out = np.array(
            [dictionary[key] for key in self.index_to_key]
        )
        self.vocabulary_ = {
            term: index for index, term in enumerate(self.feature_names_out)
        }

    def get_feature_names_out(self) -> np.ndarray:
        """Gets all feature names.

        Returns
        -------
        array of shape (n_features)
            Array of terms at each feature index.
        """
        return self.feature_names_out

    def transform(self, raw_documents: Sequence[str], y=None):
        """Transforms text into BOW matrix.

        Parameters
        ----------
        raw_documents: sequence of str
            Fixed length iterable of documents.
        y: None
            Ignored.

        Returns
        -------
        csr_array of shape (n_documents, n_features)
            Sparse bag-of-words matrix.
        """
        n_docs = len(raw_documents)
        n_features = self.feature_names_out.shape[0]
        X = spr.coo_array((n_docs, n_features), dtype=np.uint32)
        for i_document, document in enumerate(raw_documents):
            bow = self.dictionary.doc2bow(document)
            for key, count in bow:
                X[i_document, self.key_to_index[key]] = count
        return spr.csr_array(X)

    def fit(self, raw_documents: Sequence[str], y=None):
        """Does not do anything, kept for compatiblity reasons."""
        warnings.warn(
            "Gensim wrapper objects cannot be fitted, "
            "please fit the dictionary with Gensim's API, then wrap it."
        )
        return self

    def fit_transform(self, raw_documents: Sequence[str], y=None):
        """Does the same as transform(), kept for compatiblity reasons."""
        self.fit(raw_documents)
        return self.transform(raw_documents)


class TopicModelWrapper(BaseEstimator):
    """Wrapper for Gensim topic models.

    Parameters
    ----------
    model: LdaModel | LdaMulticore | Nmf | LsiModel
        Gensim topic model. The model should already be fitted.
    index_to_key: array of shape (n_features)
        Mapping of feature indices to Gensim term IDs.

    Attributes
    ----------
    components_: array of shape (n_components, n_features)
        Feature importances for each topic.
    """

    def __init__(
        self,
        model: LdaModel | LdaMulticore | Nmf | LsiModel,
        index_to_key: dict[int, int],
    ):
        self.model = model
        self.index_to_key = index_to_key
        self.components_ = model.get_topics()
        (self.n_components,) = self.components_.shape

    def _prepare_corpus(self, X) -> list[list[tuple[int, int]]]:
        n_docs, n_features = X.shape
        # Creating a list of empty lists for each doc
        X = spr.coo_array(X)
        corpus = [[]] * n_docs
        for i_doc, i_feature, count in zip(X.row, X.col, X.data):
            key = self.index_to_key[i_feature]
            corpus[i_doc].append((key, count))
        return corpus

    def fit(self, X, y=None):
        """Does not do anything, kept for compatiblity reasons."""
        warnings.warn(
            "Gensim wrapper objects cannot be fitted, "
            "please fit the dictionary with Gensim's API, then wrap it."
        )
        return self

    def transform(self, X) -> np.array:
        """Turns documents into topic distributions.

        Parameters
        ----------
        X: array of shape (n_documents, n_features)
            Feature matrix of documents (BOW or tf-idf).

        Returns
        -------
        csr_array of shape (n_docs, n_components)
            Sparse array of document-topic distributions.
        """
        corpus = self._prepare_corpus(X)
        n_docs = X.shape[0]
        doc_topic_distr = spr.coo_array((n_docs, self.n_components))
        for i_document, document in enumerate(corpus):
            doc_topic_vector = self.model[document]
            for i_topic, probability in doc_topic_vector:
                doc_topic_distr[i_document, i_topic] = probability
        return spr.csr_array(doc_topic_distr)

    def fit_transform(self, X, y=None):
        """Does the same as transform(), kept for compatiblity reasons."""
        self.fit(X)
        return self.transform(X)


def gensim_pipeline(
    dictionary: Dictionary, model: LdaModel | LdaMulticore | Nmf | LsiModel
) -> Pipeline:
    """Creates sklearn compatible wrapper for a Gensim topic pipeline.

    Parameters
    ----------
    dictionary: gensim.corpora.dictionary.Dictionary
        Gensim's dictionary object. Please only pass already fitted
        dictionary objects.
    model: LdaModel | LdaMulticore | Nmf | LsiModel
        Gensim topic model. The model should already be fitted.

    Returns
    -------
    Pipeline
        Sklearn pipeline wrapping the Gensim topic model.
    """
    vectorizer = DictionaryVectorizer(dictionary)
    topic_model = TopicModelWrapper(model=model, index_to_key=vectorizer.index_to_key)
    return make_pipeline(vectorizer, topic_model)
