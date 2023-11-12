from typing import Any, Callable, Iterable, List, Optional
from warnings import warn

import numpy as np
from dash_extensions.enrich import DashBlueprint, html
from sklearn.pipeline import Pipeline, make_pipeline

from topicwizard.pipeline import split_pipeline
from topicwizard.prepare.topics import infer_topic_names
from topicwizard.prepare.utils import get_vocab, prepare_transformed_data

BlueprintCreator = Callable[..., DashBlueprint]


def create_blank_page(name: str) -> DashBlueprint:
    blueprint = DashBlueprint()
    blueprint.layout = html.Div(id=f"{name}_container")
    return blueprint


def prepare_blueprint(
    pipeline: Pipeline,
    corpus: Iterable[str],
    create_blueprint: BlueprintCreator,
    document_names: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
    group_labels: Optional[List[str]] = None,
    *args,
    **kwargs,
) -> DashBlueprint:
    corpus = list(corpus)
    n_documents = len(corpus)
    if document_names is None:
        document_names = [f"Document {i}" for i in range(n_documents)]
    vectorizer, topic_model = split_pipeline(None, None, pipeline)
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
        if group_labels:
            group_labels = list(np.array(group_labels)[~nan_documents])
    if topic_names is None:
        topic_names = infer_topic_names(pipeline=make_pipeline(vectorizer, topic_model))
    blueprint = create_blueprint(
        vocab=vocab,
        document_term_matrix=document_term_matrix,
        document_topic_matrix=document_topic_matrix,
        topic_term_matrix=topic_term_matrix,
        document_names=document_names,
        corpus=corpus,
        topic_names=topic_names,
        pipeline=pipeline,
        group_labels=group_labels,
        *args,
        **kwargs,
    )
    return blueprint
