from __future__ import annotations

from datetime import datetime
from typing import Callable, Literal, Optional, Protocol, TypedDict

import numpy as np
from turftopic.hierarchical import TopicNode

TopicDataAttribute = Literal[
    "corpus",
    "vocab",
    "document_term_matrix",
    "document_topic_matrix",
    "topic_term_matrix",
    "document_representation",
    "transform",
    "topic_names",
    "topic_positions",
    "word_positions",
    "document_positions",
    "time_bin_edges",
    "temporal_components",
    "temporal_importance",
    "has_negative_side",
    "hierarchy",
]


class TopicData(TypedDict):
    """Inference data used to produce visualizations
    in the application and figures.

    Parameters
    ----------
    vocab: ndarray of shape (n_vocab,)
        Array of all words in the vocabulary of the topic model.
    document_term_matrix: ndarray of shape (n_documents, n_vocab)
        Bag-of-words document representations.
        Elements of the matrix are word importances/frequencies for given documents.
    document_topic_matrix: ndarray of shape (n_documents, n_topics)
        Topic importances for each document.
    topic_term_matrix: ndarray of shape (n_topics, n_vocab)
        Importances of each term for each topic in a matrix.
    document_representation: ndarray of shape (n_documents, n_dimensions)
        Embedded representations for documents.
        Can also be a sparse BoW matrix for classical models.
    topic_names: list of str, default None
        Names or topic descriptions inferred for topics by the model.
    classes: np.ndarray, default None
        Topic IDs that might be different from 0-n_topics.
        (For instance if you have an outlier topic, which is labelled -1)
    corpus: list of str, default None
        The corpus on which inference was run. Can be None.
    transform: (list[str]) -> ndarray, default None
        Function that transforms documents to document-topic matrices.
        Can be None in the case of transductive models.
    time_bin_edges: list[datetime], default None
        Edges of the time bins in a dynamic topic model.
    temporal_components: np.ndarray (n_slices, n_topics, n_vocab), default None
        Topic-term importances over time. Only relevant for dynamic topic models.
    temporal_importance: np.ndarray (n_slices, n_topics), default None
        Topic strength signal over time. Only relevant for dynamic topic models.
    has_negative_side: bool, default False
        Indicates whether the topic model's components are supposed to be interpreted in both directions.
        e.g. in SemanticSignalSeparation, one is supposed to look at highest, but also lowest ranking words.
        This is in contrast to KeyNMF for instance, where only positive word importance should be considered.
    hierarchy: TopicNode, default None
        Optional topic hierarchy for models that support hierarchical topic modeling.
    """

    vocab: np.ndarray
    document_term_matrix: np.ndarray
    document_topic_matrix: np.ndarray
    topic_term_matrix: np.ndarray
    document_representation: np.ndarray
    topic_names: Optional[list[str]]
    classes: Optional[np.ndarray]
    corpus: Optional[list[str]]
    transform: Optional[Callable]
    time_bin_edges: Optional[list[datetime]]
    temporal_components: Optional[np.ndarray]
    temporal_importance: Optional[np.ndarray]
    has_negative_side: bool
    hierarchy: Optional[TopicNode]
