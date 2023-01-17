"""Utilities for preparing data for visualization."""
from typing import Tuple
import pandas as pd

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


def topic_positions(
    topic_term_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates topic positions from topic-term matrices.

    Parameters
    ----------
    topic_term_matrix: array of shape (n_topics, n_terms)
        Topic-term matrix.

    Returns
    -------
    x: array of shape (n_topics)
    y: array of shape (n_topics)
    """
    # Calculating distances
    topic_distances = pairwise_distances(
        topic_term_matrix, metric="correlation"
    )
    n_topics = topic_term_matrix.shape[0]
    # Setting perplexity to 30, or the number of topics minus one
    perplexity = np.min((30, n_topics - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        metric="precomputed",
    )
    x, y = tsne.fit_transform(topic_distances).T
    return x, y


def topic_importances(
    topic_term_matrix: np.ndarray,
    document_term_matrix: np.ndarray,
    document_topic_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates empirical topic importances, term importances and term-topic importances.

    Parameters
    ----------
    topic_term_matrix: array of shape (n_topics, n_terms)
    document_term_matrix: array of shape (n_documents, n_terms)
    document_topic_matrix: array of shape (n_documents, n_topics)

    Returns
    -------
    topic_importances: array of shape (n_topics, )
    term_importances: array of shape (n_terms, )
    topic_term_importances: array of shape (n_topics, n_terms)
    """
    # Calculating document lengths
    document_lengths = document_term_matrix.sum(axis=1)
    # Calculating an estimate of empirical topic frequencies
    topic_importances = (document_topic_matrix.T * document_lengths).sum(
        axis=1
    )
    topic_importances = np.squeeze(np.asarray(topic_importances))
    # Calculating empirical estimate of term-topic frequencies
    topic_term_importances = (topic_term_matrix.T * topic_importances).T
    # Empirical term frequency
    term_importances = topic_term_importances.sum(axis=0)
    term_importances = np.squeeze(np.asarray(term_importances))
    return topic_importances, term_importances, topic_term_importances


def word_relevance(
    topic_id: int,
    term_frequency: np.ndarray,
    topic_term_frequency: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Returns relevance scores for each topic for each word.

    Parameters
    ----------
    components: ndarray of shape (n_topics, n_vocab)
        Topic word probability matrix.
    alpha: float
        Weight parameter.

    Returns
    -------
    ndarray of shape (n_topics, n_vocab)
        Topic word relevance matrix.
    """
    probability = np.log(topic_term_frequency[topic_id])
    probability[probability == -np.inf] = np.nan
    lift = np.log(topic_term_frequency[topic_id] / term_frequency)
    lift[lift == -np.inf] = np.nan
    relevance = alpha * probability + (1 - alpha) * lift
    return relevance


def calculate_top_words(
    topic_id: int,
    top_n: int,
    alpha: float,
    term_frequency: np.ndarray,
    topic_term_frequency: np.ndarray,
    vocab: np.ndarray,
) -> pd.DataFrame:
    """Arranges top N words by relevance for the given topic into a DataFrame."""
    vocab = np.array(vocab)
    term_frequency = np.array(term_frequency)
    topic_term_frequency = np.array(topic_term_frequency)
    relevance = word_relevance(
        topic_id, term_frequency, topic_term_frequency, alpha=alpha
    )
    highest = np.argpartition(-relevance, top_n)[:top_n]
    res = pd.DataFrame(
        {
            "word": vocab[highest],
            "importance": topic_term_frequency[topic_id, highest],
            "overall_importance": term_frequency[highest],
            "relevance": relevance[highest],
        }
    )
    return res
