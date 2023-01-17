"""Utilities for preparing data about words."""
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


def calculate_word_distances(
    topic_term_matrix: np.ndarray,
) -> np.ndarray:
    """Calculates pairwise word distances with correlation in
    topic coefficients as the metric.

    Parameters
    ----------
    topic_term_matrix: array of shape (n_topics, n_terms)

    Returns
    -------
    array of shape (n_terms, n_terms)
        Word distance matrix.
    """
    # We use the parameters of the topic model as word embeddings
    term_topic_matrix = topic_term_matrix.T
    # Calculating word distances using correlation
    word_distances = pairwise_distances(
        term_topic_matrix, metric="correlation"
    )
    return word_distances


def word_positions(
    word_distances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates word positions with manifold learning.

    Returns
    -------
    word_distances: array of shape (n_terms, n_terms)
        Word distance matrix.

    Returns
    -------
    x: array of shape (n_topics)
    y: array of shape (n_topics)
    """
    # Getting number of words in the vocabulary
    n_vocab = word_distances.shape[0]
    # Choosing perplexity such that the pipeline never fails
    perplexity = np.min((40, n_vocab - 1))
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
    selected_words: List[int],
    associated_words: List[int],
    top_n: int,
    topic_term_matrix: np.ndarray,
    topic_names: List[str],
) -> pd.DataFrame:
    """Arranges top N topics into a DataFrame for given words.
    If the number of topics is smaller than N, all topics are given back.
    """
    overall_importances = topic_term_matrix.sum(axis=1)
    overall_importances = np.squeeze(np.asarray(overall_importances))
    topic_importances = topic_term_matrix[:, selected_words].sum(axis=1)
    topic_importances = np.squeeze(np.asarray(topic_importances))
    all_words = selected_words + associated_words
    associated_importances = topic_term_matrix[:, all_words].sum(axis=1)
    associated_importances = np.squeeze(np.asarray(associated_importances))
    n_topics = topic_importances.shape[0]
    if n_topics < top_n:
        highest = np.argsort(-topic_importances)
    else:
        highest = np.argpartition(-topic_importances, top_n)[:top_n]
    # Converting List to a Series so that I can index it with a numpy array
    names = pd.Series(topic_names)
    res = pd.DataFrame(
        {
            "topic": names[highest],
            "importance": topic_importances[highest],
            "associated_importance": associated_importances[highest],
            "overall_importance": overall_importances[highest],
        }
    )
    return res


def associated_words(
    selected_words: List[int], word_distances: np.ndarray, n_association: int
) -> List[int]:
    """Returns words that are closely associated with the selected ones."""
    # Partitions array so that the smallest k elements along axis 1 are at the
    # lowest k dimensions, then I slice the array to only get the top indices
    # We do plus 1, as obviously the closest word is gonna be the word itself
    closest = np.argpartition(word_distances, kth=n_association + 1, axis=1)[
        :, 1 : n_association + 1
    ]
    associations = np.ravel(closest[selected_words])
    association_set = set(associations) - set(selected_words)
    return list(association_set)
