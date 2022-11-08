"""Module containing plotting utilities for topics."""
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def topic_plot(top_words: pd.DataFrame):
    """Plots word importances for currently selected topic."""
    top_words = top_words.sort_values("relevance", ascending=True)
    topic_word_trace = go.Bar(
        name="Estimated frequency in topic",
        y=top_words.word,
        x=top_words.importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        marker_color="#dc2626",
    )
    overall_word_trace = go.Bar(
        name="Overall frequency",
        y=top_words.word,
        x=top_words.overall_importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        marker_color="#f87171",
        textposition="outside",
        texttemplate=top_words.word,
    )
    fig = go.Figure(data=[overall_word_trace, topic_word_trace])
    fig.update_layout(
        barmode="overlay",
        plot_bgcolor="#f8fafc",
        hovermode=False,
        uniformtext=dict(
            minsize=10,
            mode="show",
        ),
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
    )
    fig.update_xaxes(
        range=[0, top_words.overall_importance.max() * 1.3],
        showticklabels=False,
    )
    fig.update_yaxes(ticks="", showticklabels=False)
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
        dragmode="pan",
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
    )
    fig.update_traces(
        textposition="top center", hovertemplate="", hoverinfo="none"
    )
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
