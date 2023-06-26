"""External API for creating self-contained figures for topics."""
from typing import Any, Iterable, List, Optional

import plotly.graph_objects as go
from sklearn.pipeline import Pipeline, make_pipeline

import topicwizard.plots.topics as plots
import topicwizard.prepare.topics as prepare
from topicwizard.app import split_pipeline
from topicwizard.prepare.utils import get_vocab, prepare_transformed_data


def plot_topic_map(
    corpus: Iterable[str],
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    topic_names: Optional[List[str]] = None,
) -> go.Figure:
    """Plots topics on a scatter plot based on the UMAP projections
    of their parameters into 2D space.
    """
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    if topic_names is None:
        topic_names = prepare.infer_topic_names(pipeline)
    corpus = list(corpus)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)
    x, y = prepare.topic_positions(topic_term_matrix)
    (
        topic_importances,
        _,
        _,
    ) = prepare.topic_importances(
        topic_term_matrix, document_term_matrix, document_topic_matrix
    )
    fig = plots.intertopic_map(
        x=x, y=y, topic_importances=topic_importances, topic_names=topic_names
    )
    return fig


def plot_most_relevant_words(
    topic_id: int,
    corpus: Iterable[str],
    top_n: int = 30,
    alpha: float = 1.0,
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
) -> go.Figure:
    """Plots most relevant words as a bar plots for a given topic."""
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    vocab = get_vocab(vectorizer)
    corpus = list(corpus)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)
    (
        topic_importances,
        term_importances,
        topic_term_importances,
    ) = prepare.topic_importances(
        topic_term_matrix, document_term_matrix, document_topic_matrix
    )
    top_words = prepare.calculate_top_words(
        topic_id, top_n, alpha, term_importances, topic_term_importances, vocab
    )
    return plots.topic_plot(top_words)


def plot_topic_wordcloud(
    topic_id: int,
    corpus: Iterable[str],
    top_n: int = 30,
    alpha: float = 1.0,
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
) -> go.Figure:
    """Plots most relevant words as a wrodcloud for a given topic."""
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    vocab = get_vocab(vectorizer)
    corpus = list(corpus)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)
    (
        topic_importances,
        term_importances,
        topic_term_importances,
    ) = prepare.topic_importances(
        topic_term_matrix, document_term_matrix, document_topic_matrix
    )
    top_words = prepare.calculate_top_words(
        topic_id, top_n, alpha, term_importances, topic_term_importances, vocab
    )
    return plots.wordcloud(top_words)
