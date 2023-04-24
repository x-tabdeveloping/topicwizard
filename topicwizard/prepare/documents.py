from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as spr
import umap


def dominant_topic(document_topic_matrix: np.ndarray) -> np.ndarray:
    """Calculates dominant topic for each document.

    Parameters
    ----------
    document_topic_matrix: array of shape (n_documents, n_topics)

    Returns
    -------
    array of shape (n_documents)
        Index of dominant topic for each document.
    """
    dominant_topic = np.argmax(document_topic_matrix, axis=1)
    return dominant_topic


def document_positions(
    document_term_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates document positions.

    Parameters
    ----------
    document_term_matrix: array of shape (n_documents, n_terms)

    Returns
    -------
    x: array of shape (n_topics)
    y: array of shape (n_topics)
    """
    # Calculating distances
    n_docs = document_term_matrix.shape[0]
    perplexity = np.min((40, n_docs - 1))
    manifold = umap.UMAP(
        n_components=2,
        n_neighbors=perplexity,
        metric="cosine",
    )
    x, y = manifold.fit_transform(document_term_matrix).T
    return x, y


def document_topic_importances(
    document_topic_matrix: np.ndarray,
) -> pd.DataFrame:
    """Rearranges the document topic matrix to a DataFrame.

    Parameters
    ----------
    document_term_matrix: array of shape (n_topics, n_terms)

    Returns
    -------
    DataFrame
        Table with document indices, topic indices and importances.
    """
    coo = spr.coo_array(document_topic_matrix)
    topic_doc_imp = pd.DataFrame(
        dict(doc_id=coo.row, topic_id=coo.col, importance=coo.data)
    )
    return topic_doc_imp


def calculate_timeline(
    doc_id: int,
    corpus: List[str],
    vectorizer: Any,
    topic_model: Any,
    window_size: int,
    step: int,
) -> np.ndarray:
    """Calculates topic timeline with a rolling window.

    Parameters
    ----------
    doc_id: int
        Index of the document.
    corpus: list of str
        List of all documents.
    vectorizer: Any
        Vectorizer component of the pipeline.
    topic_model: Any
        The topic model.
    window_size: int
        Size of the rolling windows.
    step: int
        Step size of the rolling window.

    Returns
    -------
    array of shape (n_windows, n_topics)
        Timeline of all topics over the entire document.
    """
    document = pd.Series(corpus[doc_id].split())
    windows = document.rolling(window_size, step=step)
    texts = (" ".join(window) for window in windows)
    word_timeline = vectorizer.transform(texts)
    topic_timeline = topic_model.transform(word_timeline)
    return topic_timeline
