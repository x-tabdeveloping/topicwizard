"""Utilities for preparing data for visualization."""
from typing import List, Tuple

import numpy as np
import pandas as pd
import umap
from sklearn.pipeline import Pipeline


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
    n_topics = topic_term_matrix.shape[0]
    # Setting perplexity to 30, or the number of topics minus one
    perplexity = np.min((30, n_topics - 1))
    if n_topics <= 3:
        init = "random"
    else:
        init = "spectral"
    manifold = umap.UMAP(
        n_components=2, n_neighbors=perplexity, metric="cosine", init=init
    )
    x, y = manifold.fit_transform(topic_term_matrix).T
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
    topic_importances = (document_topic_matrix.T * document_lengths).sum(axis=1)
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


def infer_topic_names(pipeline: Pipeline, top_n: int = 4) -> List[str]:
    """Infers names of topics from a trained topic model's components.
    This method does not take empirical counts or relevance into account, therefore
    automatically assigned topic names can be of low quality.

    Parameters
    ----------
    pipeline: Pipeline
        Sklearn compatible topic pipeline.
    top_n: int, default 4
        Number of words used to name the topic.

    Returns
    -------
    list of str
        List of topic names.
    """
    ((_, vectorizer), (_, topic_model)) = pipeline.steps
    components = topic_model.components_
    vocab = vectorizer.get_feature_names_out()
    highest = np.argpartition(-components, top_n)[:, :top_n]
    top_words = vocab[highest]
    topic_names = []
    for i_topic, words in enumerate(top_words):
        name = "_".join(words)
        topic_names.append(f"{i_topic}_{name}")
    return topic_names
