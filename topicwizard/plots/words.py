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
    word_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=word_frequencies,
            sizemode="area",
            sizeref=2.0 * max(word_frequencies) / (100.0**2),
            sizemin=4,
            color="rgb(168,162,158)",
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
        dragmode="pan",
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


def all_words_plot(words: pd.DataFrame) -> go.Figure:
    """Plots all word on a scatter plot based on their reduced embeddings"""
    fig = px.scatter(
        words,
        x="x",
        y="y",
        size="frequency",
        color="topic",
        size_max=80,
        hover_data=dict(word=True, x=False, y=False),
    )
    fig.update_layout(
        clickmode="event",
        modebar_remove=["lasso2d", "select2d"],
        hovermode="closest",
        plot_bgcolor="white",
        dragmode="pan",
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
    )
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


# def plot_semantic_kernel(words: pd.DataFrame, word_id: int) -> go.Figure:
# """Plots semantic kernel for the given selected word on a network graph"""
# nodes, edges = semantic_kernel(words, word_id=word_id)
# edge_x, edge_y = get_edge_positions(edges, x=nodes.x, y=nodes.y)
# node_trace = go.Scatter(
# x=nodes.x, y=nodes.y, mode="markers", text=nodes.word
# )
## TODO: finish this
# pass
