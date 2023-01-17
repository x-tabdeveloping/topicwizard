import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def word_map(
    x: np.ndarray,
    y: np.ndarray,
    word_frequencies: np.ndarray,
    vocab: np.ndarray,
) -> go.Figure:
    """Plots all words in relation to each other."""
    n_words = vocab.shape[0]
    customdata = np.array([np.arange(n_words), vocab]).T
    word_trace = go.Scattergl(
        x=x,
        y=y,
        mode="text+markers",
        text=[""] * n_words,
        marker=dict(
            size=word_frequencies,
            sizemode="area",
            sizeref=2.0 * max(word_frequencies) / (100.0**2),
            sizemin=4,
            color="#a8a29e",
        ),
        customdata=customdata,
        hovertemplate="%{customdata[1]}",
        name="",
    )
    fig = go.Figure([word_trace])
    fig.update_layout(
        clickmode="event",
        modebar_remove=["lasso2d", "select2d"],
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="white",
        # dragmode="pan",
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


def word_topics_plot(top_topics: pd.DataFrame) -> go.Figure:
    """Plots word importances for currently selected topic."""
    top_topics = top_topics.sort_values("importance", ascending=True)
    topic_word_trace = go.Bar(
        name="Importance for selected words",
        y=top_topics.topic,
        x=top_topics.importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        marker_color="#15AABF",
    )
    associated_word_trace = go.Bar(
        name="Importance with associated words",
        y=top_topics.topic,
        x=top_topics.associated_importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        marker_color="#b8eadb",
        textposition="outside",
        texttemplate=top_topics.topic,
    )
    overall_word_trace = go.Bar(
        name="Overall topic importance",
        y=top_topics.topic,
        x=top_topics.overall_importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        marker_color="rgb(168,162,158)",
        opacity=0.3,
    )
    fig = go.Figure(data=[associated_word_trace, topic_word_trace])
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
    fig.update_xaxes(
        range=[0, top_topics.associated_importance.max() * 1.3],
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
