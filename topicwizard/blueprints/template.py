from typing import Any, Callable, Iterable, List, Optional
from warnings import warn

import numpy as np
from dash_extensions.enrich import DashBlueprint

from topicwizard.prepare.utils import get_vocab, prepare_transformed_data

BlueprintCreator = Callable[..., DashBlueprint]


def prepare_blueprint(
    vectorizer: Any,
    topic_model: Any,
    corpus: Iterable[str],
    create_blueprint: BlueprintCreator,
    document_names: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
) -> DashBlueprint:
    corpus = list(corpus)
    n_documents = len(corpus)
    if document_names is None:
        document_names = [f"Document {i}" for i in range(n_documents)]
    vocab = get_vocab(vectorizer)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)
    nan_documents = np.isnan(document_topic_matrix).any(axis=1)
    n_nan_docs = np.sum(nan_documents)
    if n_nan_docs:
        warn(
            f"{n_nan_docs} documents had nan values in the output of the topic model,"
            " these are removed in preprocessing and will not be visible in the app."
        )
        corpus = list(np.array(corpus)[~nan_documents])
        document_topic_matrix = document_topic_matrix[~nan_documents]
        document_term_matrix = document_term_matrix[~nan_documents]
        document_names = list(np.array(document_names)[~nan_documents])
    n_topics = topic_term_matrix.shape[0]
    if topic_names is None:
        topic_names = [f"Topic {i}" for i in range(n_topics)]
    blueprint = create_blueprint(
        vocab=vocab,
        document_term_matrix=document_term_matrix,
        document_topic_matrix=document_topic_matrix,
        topic_term_matrix=topic_term_matrix,
        document_names=document_names,
        corpus=corpus,
        vectorizer=vectorizer,
        topic_model=topic_model,
        topic_names=topic_names,
    )
    return blueprint
