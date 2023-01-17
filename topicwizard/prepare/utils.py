from typing import Any, Iterable, Tuple

import numpy as np


def get_vocab(vectorizer: Any) -> np.ndarray:
    return vectorizer.get_feature_names_out()


def prepare_transformed_data(
    vectorizer: Any, topic_model: Any, corpus: Iterable[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transforms corpus with the topic model, and extracts important matrices.

    Parameters
    ----------
    vectorizer: Vectorizer
        Sklearn compatible text vectorizer.
    topic_model: TopicModel
        Sklearn compatible topic model.
    corpus: iterable of str
        The corpus we want to investigate the model with.

    Returns
    -------
    document_term_matrix: array of shape (n_documents, n_terms)
    document_topic_matrix: array of shape (n_documents, n_topics)
    topic_term_matrix: array of shape (n_topics, n_terms)
    """
    document_term_matrix = vectorizer.transform(corpus)
    document_topic_matrix = topic_model.transform(document_term_matrix)
    topic_term_matrix = topic_model.components_
    return document_term_matrix, document_topic_matrix, topic_term_matrix
