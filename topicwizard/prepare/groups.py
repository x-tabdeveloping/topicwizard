"""Utilities for preparing data for visualization."""
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as spr
import umap


def group_positions(
    group_topic_importances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates group positions from group-topic matrices.

    Parameters
    ----------
    group_topic_importances: array of shape (n_groups, n_topics)
        Group-topic matrix.

    Returns
    -------
    x: array of shape (n_groups)
    y: array of shape (n_groups)
    """
    # Calculating distances
    n_topics = group_topic_importances.shape[0]
    # Setting perplexity to 30, or the number of topics minus one
    perplexity = np.min((30, n_topics - 1))
    if n_topics >= 3:
        manifold = umap.UMAP(
            n_components=2, n_neighbors=perplexity, metric="cosine", init="random"
        )
        x, y = manifold.fit_transform(group_topic_importances).T
    else:
        x = np.arange(n_topics)
        y = np.zeros(n_topics)
    return x, y


def group_importances(
    document_topic_matrix: np.ndarray,
    document_term_matrix: np.ndarray,
    group_labels: np.ndarray,
    n_groups: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates empirical group importances, group-term importances
    and group-topic importances.

    Parameters
    ----------

    Returns
    -------
    group_importances: array of shape (n_groups, )
    group_term_importance: array of shape (n_groups, n_terms)
    group_topic_importances: array of shape (n_groups, n_topics)
    """
    n_terms = document_term_matrix.shape[1]
    n_topics = document_topic_matrix.shape[1]
    group_importances = np.zeros(n_groups)
    # Counting how many documents belong to each group
    for i_group in range(n_groups):
        group_importances[i_group] = np.sum(group_labels == i_group)
    # Group term importances are calculated by adding up all document bow representations
    # in a group.
    group_term_importances = np.zeros((n_groups, n_terms))
    for i_group in range(n_groups):
        group_term_importances[i_group, :] = document_term_matrix[
            group_labels == i_group
        ].sum(axis=0)
    # Group-topic importances are the same except we add up the topic representations
    group_topic_importances = np.zeros((n_groups, n_topics))
    for i_group in range(n_groups):
        group_topic_importances[i_group, :] = document_topic_matrix[
            group_labels == i_group
        ].sum(axis=0)
    return group_importances, group_term_importances, group_topic_importances


def dominant_topic(group_topic_importances: np.ndarray) -> np.ndarray:
    """Calculates dominant topic for each group.

    Parameters
    ----------
    group_topic_importances: array of shape (n_groups, n_topics)

    Returns
    -------
    array of shape (n_documents)
        Index of dominant topic for each group.
    """
    dominant_topic = np.argmax(group_topic_importances, axis=1)
    return dominant_topic


def top_topics(
    group_id: int,
    top_n: int,
    group_topic_importances: np.ndarray,
    topic_names: List[str],
) -> pd.DataFrame:
    """Finds top topics for a given group and arranges them into a DataFrame."""
    overall_importances = group_topic_importances.sum(axis=0)
    overall_importances = np.squeeze(np.asarray(overall_importances))
    topic_importances = group_topic_importances[group_id]
    topic_importances = np.squeeze(np.asarray(topic_importances))
    n_topics = topic_importances.shape[0]
    if n_topics <= top_n:
        highest = np.argsort(-topic_importances)
    else:
        highest = np.argpartition(-topic_importances, top_n)[:top_n]
    # Converting List to a Series so that I can index it with a numpy array
    names = pd.Series(topic_names)
    res = pd.DataFrame(
        {
            "topic": names[highest],
            "topic_id": highest,
            "importance": topic_importances[highest],
            "overall_importance": overall_importances[highest],
        }
    )
    return res


def top_words(
    group_id: int,
    top_n: int,
    group_term_importances: np.ndarray,
    vocab: np.ndarray,
) -> pd.DataFrame:
    """Finds top words for a given group."""
    vocab = np.array(vocab)
    importances = group_term_importances[group_id]
    importances = np.squeeze(np.asarray(importances))
    overall_importances = group_term_importances.sum(axis=0)
    overall_importances = np.squeeze(np.asarray(overall_importances))
    n_vocab = vocab.shape[0]
    if n_vocab <= top_n:
        highest = np.argsort(-importances)
    else:
        highest = np.argpartition(-importances, top_n)[:top_n]
    res = pd.DataFrame(
        {
            "word": vocab[highest],
            "importance": importances[highest],
            "overall_importance": overall_importances[highest],
        }
    )
    return res
