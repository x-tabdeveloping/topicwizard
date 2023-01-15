"""Utilities for preparing data about words."""
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


def word_positions(
    topic_term_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates word positions with manifold learning.
    Uses correlation as a distance metric.

    Parameters
    ----------
    topic_term_matrix: array of shape (n_topics, n_terms)
        Topic-term matrix.

    Returns
    -------
    x: array of shape (n_topics)
    y: array of shape (n_topics)
    """
    # Getting number of words in the vocabulary
    n_vocab = topic_term_matrix.shape[1]
    # We use the parameters of the topic model as word embeddings
    term_topic_matrix = topic_term_matrix.T
    # Calculating word distances using correlation
    word_distances = pairwise_distances(
        term_topic_matrix, metric="correlation"
    )
    # Choosing perplexity such that the pipeline never fails
    perplexity = np.min((30, n_vocab - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        metric="precomputed",
    )
    x, y = tsne.fit_transform(word_distances).T
    return x, y


def word_importances(
    document_term_matrix: np.ndarray,
) -> np.ndarray:
    """Calculates overall word frequencies.

    Parameters
    ----------
    document_term_matrix: array of shape (n_documents, n_terms)

    Returns
    -------
    array of shape (n_terms, )
    """
    word_frequencies = document_term_matrix.sum(axis=0)
    word_frequencies = np.squeeze(np.asarray(word_frequencies))
    return word_frequencies


def top_topics(
    term_id: int,
    top_n: int,
    topic_term_matrix: np.ndarray,
    topic_names: List[str],
) -> pd.DataFrame:
    """Arranges top N topics into a DataFrame for a given word.
    If the number of topics is smaller than N, all topics are given back.
    """
    topic_importances = topic_term_matrix[:, term_id]
    topic_importances = np.squeeze(np.asarray(topic_importances))
    n_topics = topic_importances.shape[0]
    if n_topics < top_n:
        highest = np.argsort(-topic_importances)
    else:
        highest = np.argpartition(-topic_importances, top_n)[:top_n]
    # Converting List to a Series so that I can index it with a numpy array
    names = pd.Series(topic_names)
    res = pd.DataFrame(
        {
            "word": names[highest],
            "importance": topic_importances[highest],
        }
    )
    return res
