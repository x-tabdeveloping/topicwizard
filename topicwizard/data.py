from typing import (Callable, Dict, Iterable, List, Literal, Optional,
                    TypedDict, Union)
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
