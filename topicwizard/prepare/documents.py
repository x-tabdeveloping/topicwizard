from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as spr

from topicwizard.prepare.dimensionality_reduction import reduce_manifold_2d


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
    document_term_matrix: array of shape (n_topics, n_terms)

    Returns
    -------
    x: array of shape (n_topics)
    y: array of shape (n_topics)
    """
    x, y = reduce_manifold_2d(document_term_matrix, which="umap").T
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
