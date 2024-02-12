"""External API for creating self-contained figures for groups."""
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import colors
from plotly.subplots import make_subplots

import topicwizard.plots.groups as plots
import topicwizard.prepare.groups as prepare
from topicwizard.data import TopicData


def group_map(topic_data: TopicData, group_labels: List[str]) -> go.Figure:
    """Projects groups into 2d space and displays them on a scatter plot.

    Parameters
    ----------
    topic_data: TopicData
        Inference data from topic modeling.
    group_labels: list[str]
        Labels for each of the documents in the corpus.
    """
    # Factorizing group labels
    group_id_labels, group_names = pd.factorize(group_labels)
    n_groups = group_names.shape[0]
    (
        group_importances,
        group_term_importances,
        group_topic_importances,
    ) = prepare.group_importances(
        topic_data["document_topic_matrix"],
        topic_data["document_term_matrix"],
        group_id_labels,
        n_groups,
    )
    x, y = prepare.group_positions(group_term_importances)
    dominant_topic = prepare.dominant_topic(group_topic_importances)
    dominant_topic = np.array(topic_data["topic_names"])[dominant_topic]
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
    topic_data: TopicData, group_labels: List[str], top_n: int = 5, n_columns: int = 4
):
    """Displays the most important topics for each group.

    Parameters
    ----------
    topic_data: TopicData
        Inference data from topic modeling.
    group_labels: list[str]
        Labels for each of the documents in the corpus.
    top_n: int, default 5
        Maximum number of topics to display for each group.
    n_columns: int, default 4
        Indicates how many columns the faceted plot should have.
    """
    # Factorizing group labels
    group_id_labels, group_names = pd.factorize(group_labels)
    n_groups = group_names.shape[0]
    (
        group_importances,
        group_term_importances,
        group_topic_importances,
    ) = prepare.group_importances(
        topic_data["document_topic_matrix"],
        topic_data["document_term_matrix"],
        group_id_labels,
        n_groups,
    )
    n_rows = (n_groups // n_columns) + 1
    fig = make_subplots(
        rows=n_rows,
        cols=n_columns,
        subplot_titles=group_names,
        vertical_spacing=0.03,
        horizontal_spacing=0.01,
    )
    n_topics = len(topic_data["topic_names"])
    color_scheme = colors.get_colorscale("Portland")
    topic_colors = colors.sample_colorscale(
        color_scheme, np.arange(n_topics) / n_topics, low=0.25, high=1.0
    )
    topic_colors = np.array(topic_colors)
    # Here I am collecting the maximal importance for each group,
    # So that the x axis can be adjusted to this.
    for group_id in range(n_groups):
        top_topics = prepare.top_topics(
            group_id, top_n, group_topic_importances, topic_data["topic_names"]
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
    topic_data: TopicData, group_labels: List[str], top_n: int = 30, n_columns: int = 4
) -> go.Figure:
    """Plots wordclouds for each group.

    Parameters
    ----------
    topic_data: TopicData
        Inference data from topic modeling.
    group_labels: list[str]
        Labels for each document in the corpus.
    top_n: int, default 30
        Number of words to display for each group.
    n_columns: int, default 4
        Number of columns the faceted plot should have.
    """
    # Factorizing group labels
    group_id_labels, group_names = pd.factorize(group_labels)
    n_groups = group_names.shape[0]
    (
        group_importances,
        group_term_importances,
        group_topic_importances,
    ) = prepare.group_importances(
        topic_data["document_topic_matrix"],
        topic_data["document_term_matrix"],
        group_id_labels,
        n_groups,
    )
    n_rows = (n_groups // n_columns) + 1
    fig = make_subplots(
        rows=n_rows,
        cols=n_columns,
        subplot_titles=group_names,
        vertical_spacing=0.05,
        horizontal_spacing=0.01,
    )
    for group_id in range(n_groups):
        top_words = prepare.top_words(
            group_id, top_n, group_term_importances, topic_data["vocab"]
        )
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
