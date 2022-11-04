"""Module containing plotting utilities for topics."""
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def topic_plot(topic: int, top_words: pd.DataFrame):
    """Plots word importances for currently selected topic.

    Parameters
    ----------
    topic: int
        Index of the topic to be displayed.
    genre_importance: DataFrame
        Data about genre importances.
    top_words: DataFrame
        Data about word importances for each topic.

    Returns
    -------
    Figure
        Bar chart of word importances.
    """
    top_words = top_words[top_words.topic == topic]
    top_words = top_words.sort_values("importance", ascending=False)
    topic_word_trace = go.Bar(
        name="Importance for topic",
        y=top_words.word,
        x=top_words.importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        marker_color="#dc2626",
    )
    overall_word_trace = go.Bar(
        name="Overall importance",
        y=top_words.word,
        x=top_words.overall_importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        marker_color="#f87171",
    )
    fig = go.Figure(data=[overall_word_trace, topic_word_trace])
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        barmode="overlay",
        plot_bgcolor="#f8fafc",
    )
    return fig


def all_topics_plot(topic_data: pd.DataFrame, current_topic: int) -> go.Figure:
    """Plots all topics on a bubble plot with estimated distances and importances.

    Parameters
    ----------
    topic_data: DataFrame
        Data about topic names, positions and sizes.

    Returns
    -------
    Figure
        Bubble plot of topics.
    """
    topic_data = topic_data.assign(
        selected=(topic_data.topic_id == current_topic).astype(int)
    )
    fig = px.scatter(
        topic_data,
        x="x",
        y="y",
        color="selected",
        text="topic_name",
        custom_data=["topic_id"],
        size="size",
        color_continuous_scale="Sunset_r",
    )
    fig.update_layout(
        clickmode="event",
        modebar_remove=["lasso2d", "select2d"],
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="white",
    )
    fig.update_traces(textposition="top center", hovertemplate="")
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(
        showticklabels=False,
        title="",
        gridcolor="#e5e7eb",
        linecolor="#f9fafb",
        linewidth=6,
        mirror=True,
        zerolinewidth=2,
        zerolinecolor="#d1d5db",
    )
    fig.update_yaxes(
        showticklabels=False,
        title="",
        gridcolor="#e5e7eb",
        linecolor="#f9fafb",
        mirror=True,
        linewidth=6,
        zerolinewidth=2,
        zerolinecolor="#d1d5db",
    )
    return fig
