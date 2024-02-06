from typing import Callable, Dict, List, Literal, Optional, TypedDict, Union
from warnings import warn

import numpy as np


class TopicData(TypedDict):
    corpus: List[str]
    vocab: np.ndarray
    document_term_matrix: np.ndarray
    document_topic_matrix: np.ndarray
    topic_term_matrix: np.ndarray
    document_representation: np.ndarray
    transform: Optional[Callable]
    topic_names: List[str]


def filter_nan_docs(topic_data: TopicData) -> None:
    """Filters out documents, the topical content of which contains nans.
    NOTE: The function works in place.
    """
    nan_documents = np.isnan(topic_data["document_topic_matrix"]).any(axis=1)
    n_nan_docs = np.sum(nan_documents)
    if n_nan_docs:
        warn(
            f"{n_nan_docs} documents had nan values in the output of the topic model,"
            " these are removed in preprocessing and will not be visible in the app."
        )
        topic_data["corpus"] = list(np.array(topic_data["corpus"])[~nan_documents])
        topic_data["document_topic_matrix"] = topic_data["document_topic_matrix"][
            ~nan_documents
        ]
        topic_data["document_term_matrix"] = topic_data["document_term_matrix"][
            ~nan_documents
        ]
        topic_data["document_names"] = list(
            np.array(topic_data["document_names"])[~nan_documents]
        )
        if topic_data["group_labels"]:
            topic_data["group_labels"] = list(
                np.array(topic_data["group_labels"])[~nan_documents]
            )
