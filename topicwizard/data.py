from typing import Callable, List, Optional, TypedDict

import numpy as np


class TopicData(TypedDict):
    """Inference data used to produce visualizations
    in the application and figures.

    Attributes
    ----------
    corpus: list of str
        The corpus on which inference was run.
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
    transform: (list[str]) -> ndarray, optional
        Function that transforms documents to document-topic matrices.
        Can be None in the case of transductive models.
    topic_names: list of str
        Names or topic descriptions inferred for topics by the model.
    """

    corpus: List[str]
    vocab: np.ndarray
    document_term_matrix: np.ndarray
    document_topic_matrix: np.ndarray
    topic_term_matrix: np.ndarray
    document_representation: np.ndarray
    transform: Optional[Callable]
    topic_names: List[str]
    topic_positions: Optional[np.ndarray]
    word_positions: Optional[np.ndarray]
    document_positions: Optional[np.ndarray]
