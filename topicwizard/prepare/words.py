"""Utilities for preparing data about words."""
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline


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
    topic_term_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates word positions with manifold learning.

    Returns
    -------
    topic_term_matrix: array of shape (n_topics, n_terms)

    Returns
    -------
    x: array of shape (n_topics)
    y: array of shape (n_topics)
    """
    # Getting number of words in the vocabulary
    n_topics, n_vocab = topic_term_matrix.shape
    # We use the parameters of the topic model as word embeddings
    term_topic_matrix = topic_term_matrix.T
    # Adding pre-manifold reduction, so that it runs faster
    pca = PCA(n_components=np.min((n_topics, 10)))
    # Choosing perplexity such that the pipeline never fails
    perplexity = np.min((40, n_vocab - 1))
    manifold = TSNE(
        n_components=2,
        # affinity="nearest_neighbors",
        perplexity=perplexity,
        init="pca",
        metric="euclidean",
        learning_rate="auto",
        n_iter=400,
        n_iter_without_progress=100,
    )
    reduction_pipeline = Pipeline(
        [
            ("pca", pca),
            ("scaler", StandardScaler()),
            ("manifold", manifold),
        ]
    )
    x, y = reduction_pipeline.fit_transform(term_topic_matrix).T
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
    selected_words: List[int],
    topic_term_matrix: np.ndarray,
    n_association: int,
) -> List[int]:
    """Returns words that are closely associated with the selected ones."""
    term_topic_matrix = topic_term_matrix.T
    # Selecting terms
    selected_terms_matrix = term_topic_matrix[selected_words]
    # Calculating all distances from the selected words
    distances = pairwise_distances(
        selected_terms_matrix, term_topic_matrix, metric="euclidean"
    )
    # Partitions array so that the smallest k elements along axis 1 are at the
    # lowest k dimensions, then I slice the array to only get the top indices
    # We do plus 1, as obviously the closest word is gonna be the word itself
    closest = np.argpartition(distances, kth=n_association + 1, axis=1)[
        :, 1 : n_association + 1
    ]
    associations = np.ravel(closest)
    association_set = set(associations) - set(selected_words)
    return list(association_set)
