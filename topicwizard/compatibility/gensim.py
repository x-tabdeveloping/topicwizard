import warnings
from typing import Sequence

import numpy as np
import scipy.sparse as spr
from gensim.corpora.dictionary import Dictionary
from sklearn.base import BaseEstimator


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
        coo_array of shape (n_documents, n_features)
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
