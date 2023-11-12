"""External API for creating self-contained figures for groups."""
from typing import Any, Iterable, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import colors
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline, make_pipeline

import topicwizard.plots.groups as plots
import topicwizard.prepare.groups as prepare
from topicwizard.app import split_pipeline
from topicwizard.prepare.topics import infer_topic_names
from topicwizard.prepare.utils import get_vocab, prepare_transformed_data


def group_map(
    corpus: Iterable[str],
    group_labels: List[str],
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    topic_names: Optional[List[str]] = None,
    representation: Literal["term", "topic"] = "term",
) -> go.Figure:
    """Plots groups on a scatter plot based on the UMAP projections
    of their representations in the model into 2D space.

    Parameters
    ----------
    corpus: iterable of str
        List of all works in the corpus you intend to visualize.
    group_labels: list of str
        List of group labels for each document.
    pipeline: Pipeline, default None
        Sklearn compatible pipeline, that has two components:
        a vectorizer and a topic model.
        Ignored if vectorizer and topic_model are provided.
    vectorizer: Vectorizer, default None
        Sklearn compatible vectorizer, that turns texts into
        bag-of-words representations.
    topic_model: TopicModel, default None
        Sklearn compatible topic model, that can transform documents
        into topic distributions.
    topic_names: list of str, default None
        List of topic names in the corpus, if not provided
        topic names will be inferred.
    representation: {"term", "topic"}, default "term"
        Determines which representation of the groups should be
        projected to 2D space and displayed.
        If 'term', representations returned from the vectorizer
        will be used, if 'topic', representations returned by
        the topic model will be used. This can be particularly
        advantageous with non-bag-of-words topic models.

    Returns
    -------
    go.Figure
        Map of groups.
    """
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    if topic_names is None:
        topic_names = infer_topic_names(pipeline)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)
    # Factorizing group labels
    group_id_labels, group_names = pd.factorize(group_labels)
    n_groups = group_names.shape[0]
    (
        group_importances,
        group_term_importances,
        group_topic_importances,
    ) = prepare.group_importances(
        document_topic_matrix, document_term_matrix, group_id_labels, n_groups
    )
    if representation == "term":
        x, y = prepare.group_positions(group_term_importances)
    else:
        x, y = prepare.group_positions(group_topic_importances)
    dominant_topic = prepare.dominant_topic(group_topic_importances)
    dominant_topic = np.array(topic_names)[dominant_topic]
    groups_df = pd.DataFrame(
        dict(
            dominant_topic=dominant_topic,
            x=x,
            y=y,
            group_name=group_names,
            frequency=group_importances,
        )
    )
    return px.scatter(
        groups_df,
        x="x",
        y="y",
        color="dominant_topic",
        size="frequency",
        text="group_name",
        size_max=100,
        hover_data={
            "dominant_topic": True,
            "group_name": True,
            "frequency": True,
            "x": False,
            "y": False,
        },
        template="plotly_white",
    )


def group_topic_barcharts(
    corpus: Iterable[str],
    group_labels: List[str],
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    topic_names: Optional[List[str]] = None,
    top_n: int = 5,
    n_columns: int = 4,
) -> go.Figure:
    """Plots topic importance barcharts for all groups.

    Parameters
    ----------
    corpus: iterable of str
        List of all works in the corpus you intend to visualize.
    group_labels: list of str
        List of group labels for each document.
    pipeline: Pipeline, default None
        Sklearn compatible pipeline, that has two components:
        a vectorizer and a topic model.
        Ignored if vectorizer and topic_model are provided.
    vectorizer: Vectorizer, default None
        Sklearn compatible vectorizer, that turns texts into
        bag-of-words representations.
    topic_model: TopicModel, default None
        Sklearn compatible topic model, that can transform documents
        into topic distributions.
    topic_names: list of str, default None
        List of topic names in the corpus, if not provided
        topic names will be inferred.
    top_n: int, default 5
        Number of top topics to display for a given group.
    n_columns: int, default 4
        Number of columns allowed in the subplot grid.

    Returns
    -------
    go.Figure
        Topic importance barcharts for all groups.
    """
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    if topic_names is None:
        topic_names = infer_topic_names(pipeline)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)
    # Factorizing group labels
    group_id_labels, group_names = pd.factorize(group_labels)
    n_groups = group_names.shape[0]
    (
        group_importances,
        group_term_importances,
        group_topic_importances,
    ) = prepare.group_importances(
        document_topic_matrix, document_term_matrix, group_id_labels, n_groups
    )
    n_rows = (n_groups // n_columns) + 1
    fig = make_subplots(
        rows=n_rows,
        cols=n_columns,
        subplot_titles=group_names,
        vertical_spacing=0.03,
        horizontal_spacing=0.01,
    )
    n_topics = len(topic_names)
    color_scheme = colors.get_colorscale("Portland")
    topic_colors = colors.sample_colorscale(
        color_scheme, np.arange(n_topics) / n_topics, low=0.25, high=1.0
    )
    topic_colors = np.array(topic_colors)
    # Here I am collecting the maximal importance for each group,
    # So that the x axis can be adjusted to this.
    for group_id in range(n_groups):
        top_topics = prepare.top_topics(
            group_id, top_n, group_topic_importances, topic_names
        )
        max_importance = top_topics.overall_importance.max()
        subfig = plots.group_topics_barchart(top_topics, topic_colors=topic_colors)
        row, column = (group_id // n_columns) + 1, (group_id % n_columns) + 1
        for trace in subfig.data:
            # hiding legend if it isn't the first trace.
            if group_id:
                trace.showlegend = False
            fig.add_trace(trace, row=row, col=column)
            fig.update_xaxes(range=[0, max_importance * 1.5], row=row, col=column)
    fig.update_layout(
        barmode="overlay",
        plot_bgcolor="white",
        hovermode=False,
        uniformtext=dict(
            minsize=10,
            mode="show",
        ),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.6)",
        ),
        margin=dict(l=0, r=0, b=18, pad=2),
    )
    fig.update_xaxes(
        showticklabels=False,
    )
    fig.update_yaxes(ticks="", showticklabels=False)
    fig.update_xaxes(
        gridcolor="#e5e7eb",
    )
    fig.update_yaxes(
        gridcolor="#e5e7eb",
    )
    return fig


def group_wordclouds(
    corpus: Iterable[str],
    group_labels: List[str],
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    topic_names: Optional[List[str]] = None,
    top_n: int = 30,
    n_columns: int = 4,
) -> go.Figure:
    """Displays wordclouds for each of the groups based on word frequencies.

    Parameters
    ----------
    corpus: iterable of str
        List of all works in the corpus you intend to visualize.
    group_labels: list of str
        List of group labels for each document.
    pipeline: Pipeline, default None
        Sklearn compatible pipeline, that has two components:
        a vectorizer and a topic model.
        Ignored if vectorizer and topic_model are provided.
    vectorizer: Vectorizer, default None
        Sklearn compatible vectorizer, that turns texts into
        bag-of-words representations.
    topic_model: TopicModel, default None
        Sklearn compatible topic model, that can transform documents
        into topic distributions.
    topic_names: list of str, default None
        List of topic names in the corpus, if not provided
        topic names will be inferred.
    top_n: int, default 30
        Number of top topics to display for a given group.
    n_columns: int, default 6
        Number of columns allowed in the subplot grid.

    Returns
    -------
    go.Figure
        Topic importance barcharts for all groups.
    """
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    if topic_names is None:
        topic_names = infer_topic_names(pipeline)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)
    # Factorizing group labels
    group_id_labels, group_names = pd.factorize(group_labels)
    n_groups = group_names.shape[0]
    (
        group_importances,
        group_term_importances,
        group_topic_importances,
    ) = prepare.group_importances(
        document_topic_matrix, document_term_matrix, group_id_labels, n_groups
    )
    vocab = get_vocab(vectorizer)
    n_rows = (n_groups // n_columns) + 1
    fig = make_subplots(
        rows=n_rows,
        cols=n_columns,
        subplot_titles=group_names,
        vertical_spacing=0.05,
        horizontal_spacing=0.01,
    )
    for group_id in range(n_groups):
        top_words = prepare.top_words(group_id, top_n, group_term_importances, vocab)
        subfig = plots.wordcloud(top_words)
        row, column = (group_id // n_columns) + 1, (group_id % n_columns) + 1
        fig.add_trace(subfig.data[0], row=row, col=column)
    fig.update_layout(
        plot_bgcolor="white",
    )
    fig.update_yaxes(
        showticklabels=False,
        gridcolor="white",
        linecolor="white",
        zerolinecolor="white",
    )
    fig.update_xaxes(
        showticklabels=False,
        gridcolor="white",
        linecolor="white",
        zerolinecolor="white",
    )
    fig.update_traces(hovertemplate="", hoverinfo="none")
    return fig
