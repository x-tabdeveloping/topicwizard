"""Module containing plotting utilities for topics."""
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.preprocessing import minmax_scale
from wordcloud import WordCloud


def intertopic_map(
    x: np.ndarray,
    y: np.ndarray,
    topic_importances: np.ndarray,
    topic_names: List[str],
) -> go.Figure:
    size = 1 + minmax_scale(topic_importances)
    text = [f"<b>{name}</b>" for name in topic_names]
    topic_trace = go.Scatter(
        x=x,
        y=y,
        mode="text+markers",
        text=text,
        textfont=dict(size=14),
        marker=dict(
            size=size,
            sizemode="area",
            sizeref=2.0 * max(size) / (100.0**2),
            sizemin=4,
            color="rgb(168,162,158)",
            line=dict(width=3, color="black"),
        ),
        customdata=np.atleast_2d(np.arange(x.shape[0])).T,
    )
    fig = go.Figure([topic_trace])
    fig.update_layout(
        clickmode="event",
        modebar_remove=["lasso2d", "select2d"],
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="white",
        dragmode="pan",
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
    )
    fig.update_traces(textposition="top center", hovertemplate="", hoverinfo="none")
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


def topic_plot(top_words: pd.DataFrame):
    """Plots word importances for currently selected topic."""
    top_words = top_words.sort_values("relevance", ascending=True)
    overlap = np.any(top_words.overall_importance < top_words.importance)
    if overlap:
        print(top_words[top_words.overall_importance < top_words.importance])
    text = top_words.word.map(lambda s: f"<b>{s}</b>")
    fig = go.Figure()
    if not overlap:
        overall_word_trace = go.Bar(
            name="Summed importances over topics",
            y=top_words.word,
            x=top_words.overall_importance,
            orientation="h",
            base=dict(x=[0.5, 1]),
            marker_color="rgba(168,162,158, 0.3)",
            marker_line=dict(color="rgb(168,162,158)", width=3),
        )
        fig.add_trace(overall_word_trace)
    topic_word_trace = go.Bar(
        name="Estimated importance in topic",
        y=top_words.word,
        x=top_words.importance,
        text=text,
        textposition="auto",
        orientation="h",
        base=dict(x=[0.5, 1]),
        marker_color="rgba(251,146,60,0.65)",
        marker_line=dict(color="black", width=3),
    )
    fig.add_trace(topic_word_trace)
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
        margin=dict(l=0, r=0, b=18, t=0, pad=0),
    )
    max_overall = top_words.overall_importance.max()
    max_specific = top_words.importance.max()
    max_val = max(max_overall, max_specific)
    min_overall = top_words.overall_importance.min()
    min_specific = top_words.importance.min()
    min_val = min(min_overall, min_specific)
    lower = min_val * 1.3 if min_val < 0 else 0
    upper = max_val * 1.3
    fig.update_xaxes(
        range=[lower, upper],
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


def wordcloud(top_words: pd.DataFrame) -> go.Figure:
    """Plots most relevant words for current topic as a worcloud."""
    top_dict = {
        word: importance
        for word, importance in zip(
            top_words.word, 0.1 + minmax_scale(top_words.importance)
        )
    }
    cloud = WordCloud(
        width=800,
        height=1060,
        background_color="white",
        colormap="copper",
        scale=4,
    ).generate_from_frequencies(top_dict)
    image = cloud.to_image()
    image = image.resize((1600, 2120), resample=Image.Resampling.LANCZOS)
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
