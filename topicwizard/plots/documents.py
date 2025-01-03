"""Module containing plotting utilities for documents."""

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as spr
from PIL import Image
from wordcloud import WordCloud

from topicwizard.plots.utils import get_default_font_path


def document_map(
    x: np.ndarray,
    y: np.ndarray,
    document_names: List[str],
    dominant_topic: np.ndarray,
    topic_colors: np.ndarray,
) -> go.Figure:
    n_documents = x.shape[0]
    customdata = np.array([np.arange(n_documents), document_names]).T
    trace = go.Scattergl(
        x=x,
        y=y,
        mode="markers+text",
        text=[""] * n_documents,
        marker=dict(
            color=topic_colors[dominant_topic],
            size=15,
            opacity=0.5,
            line=dict(color="white", width=1.5),
        ),
        customdata=customdata,
        hovertemplate="%{customdata[1]}",
        name="",
        textfont=dict(size=16, color="black"),
    )
    fig = go.Figure([trace])
    fig.update_layout(
        clickmode="event",
        modebar_remove=["lasso2d", "select2d"],
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
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


def document_topic_barplot(
    topic_importances: pd.DataFrame,
    topic_names: List[str],
    topic_colors: np.ndarray,
    top_n: int = 15,
) -> go.Figure:
    name_mapping = pd.Series(topic_names)
    n_topics = len(topic_names)
    color_mapping = {topic_names[i]: topic_colors[i] for i in range(n_topics)}
    topic_importances = topic_importances.assign(
        topic_name=topic_importances.topic_id.map(name_mapping)
    )
    topic_importances = topic_importances.sort_values(
        "importance", ascending=False
    ).iloc[:top_n]
    # topic_names = [f"<b>{name}</b>" for name in topic_names]
    fig = px.bar(
        topic_importances,
        x="importance",
        y="topic_name",
        color="topic_name",
        color_discrete_map=color_mapping,
        # text="topic_name",
    )
    fig.update_traces(
        textposition="inside", marker=dict(line=dict(color="black", width=3))
    )
    fig.update_yaxes(title="Topic")
    fig.update_xaxes(title="Importance")
    fig.update_xaxes(
        gridcolor="#e5e7eb",
        linecolor="#f9fafb",
        linewidth=6,
        mirror=True,
        zerolinewidth=2,
        zerolinecolor="#d1d5db",
    )
    fig.update_yaxes(
        gridcolor="#e5e7eb",
        linecolor="#f9fafb",
        linewidth=6,
        mirror=True,
        zerolinewidth=2,
        zerolinecolor="#d1d5db",
    )
    fig.update_layout(
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig


def document_topic_plot(
    topic_importances: pd.DataFrame,
    topic_names: List[str],
    topic_colors: np.ndarray,
) -> go.Figure:
    """Plots topic importances for a selected document.

    Parameters
    ----------
    topic_importances: dict of int to float
        Mapping of topic id's to importances.
    topic_names: list of str
        List of topic names.

    Returns
    -------
    Figure
        Pie chart of topic importances for each document.
    """
    name_mapping = pd.Series(topic_names)
    n_topics = len(topic_names)
    color_mapping = {topic_names[i]: topic_colors[i] for i in range(n_topics)}
    topic_importances = topic_importances.assign(
        topic_name=topic_importances.topic_id.map(name_mapping)
    )
    fig = px.pie(
        topic_importances,
        values="importance",
        names="topic_name",
        color="topic_name",
        color_discrete_map=color_mapping,
        # color_discrete_sequence=px.colors.cyclical.Twilight,
    )
    fig.update_traces(textposition="inside", textinfo="label")
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(1,1,1,0)",
        plot_bgcolor="rgba(1,1,1,0)",
    )
    return fig


def document_timeline(
    topic_timeline: np.ndarray, topic_names: List[str], topic_colors: np.ndarray
) -> go.Figure:
    topic_timeline = topic_timeline.T
    traces = []
    n_topics = len(topic_names)
    for topic_id in range(n_topics):
        timeline = topic_timeline[topic_id]
        timeline = np.squeeze(np.asarray(timeline))
        trace = go.Scattergl(
            x=np.arange(timeline.shape[0]),
            y=timeline,
            mode="lines",
            name=topic_names[topic_id],
            marker=dict(color=topic_colors[topic_id]),
        )
        traces.append(trace)
    fig = go.Figure(data=traces)
    fig.update_layout(
        hovermode="closest",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend_title="Topic",
    )
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(
        title="Window",
        gridcolor="#e5e7eb",
        linecolor="#f9fafb",
        linewidth=6,
        mirror=True,
        zerolinewidth=2,
        zerolinecolor="#d1d5db",
    )
    fig.update_yaxes(
        title="Importance",
        gridcolor="#e5e7eb",
        linecolor="#f9fafb",
        mirror=True,
        linewidth=6,
        zerolinewidth=2,
        zerolinecolor="#d1d5db",
    )
    return fig


def document_wordcloud(
    doc_id: int,
    document_term_matrix: np.ndarray,
    vocab: np.ndarray,
    custom_font_path=None,
) -> go.Figure:
    coo = spr.coo_array(document_term_matrix[[doc_id], :])
    term_dict = {vocab[column]: data for column, data in zip(coo.col, coo.data)}
    font_path = custom_font_path or get_default_font_path().absolute()
    cloud = WordCloud(
        font_path=font_path,
        width=800,
        height=800,
        background_color="white",
        colormap="twilight",
        scale=4,
    ).generate_from_frequencies(term_dict)
    image = cloud.to_image()
    image = image.resize((1600, 1600), resample=Image.Resampling.LANCZOS)
    fig = px.imshow(image)
    fig.update_layout(
        dragmode="pan",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
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
