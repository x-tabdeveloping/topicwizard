"""External API for creating self-contained figures for documents."""
from typing import Any, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
import scipy.sparse as spr
from plotly import colors
from sklearn.pipeline import Pipeline, make_pipeline

import topicwizard.plots.documents as plots
import topicwizard.prepare.documents as prepare
from topicwizard.app import split_pipeline
from topicwizard.prepare.topics import infer_topic_names
from topicwizard.prepare.utils import get_vocab, prepare_transformed_data


def plot_document_topic_distribution(
    documents: Union[List[str], str],
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    topic_names: Optional[List[str]] = None,
) -> go.Figure:
    """Plots the distribution of topics in the given documents."""
    if isinstance(documents, str):
        documents = [documents]
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    if topic_names is None:
        topic_names = infer_topic_names(pipeline)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, documents)
    topic_importances = prepare.document_topic_importances(document_topic_matrix)
    topic_importances = topic_importances.groupby(["topic_id"]).sum().reset_index()
    n_topics = document_topic_matrix.shape[-1]
    twilight = colors.get_colorscale("Twilight")
    topic_colors = colors.sample_colorscale(twilight, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    return plots.document_topic_plot(
        topic_importances,
        topic_names,
        topic_colors,
    )


def plot_document_wordcloud(
    documents: Union[List[str], str],
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
) -> go.Figure:
    """Plots word importances for the given documents as a wordcloud."""
    if isinstance(documents, str):
        documents = [documents]
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    vocab = get_vocab(vectorizer)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, documents)
    document_term_matrix = spr.coo_array(document_term_matrix).sum(axis=0)
    return plots.document_wordcloud(
        doc_id=0, document_term_matrix=document_term_matrix, vocab=vocab
    )


def plot_document_topic_timeline(
    document: str,
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    topic_names: Optional[List[str]] = None,
    window_size: int = 10,
    step: int = 1,
) -> go.Figure:
    """Plots timeline of topics inside a single document."""
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    if topic_names is None:
        topic_names = infer_topic_names(pipeline)
    timeline = prepare.calculate_timeline(
        doc_id=0,
        corpus=[document],
        vectorizer=vectorizer,
        topic_model=topic_model,
        window_size=window_size,
        step=step,
    )
    n_topics = len(topic_names)
    twilight = colors.get_colorscale("Twilight")
    topic_colors = colors.sample_colorscale(twilight, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    return plots.document_timeline(timeline, topic_names, topic_colors)
