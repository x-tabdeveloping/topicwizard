"""External API for creating self-contained figures for topics."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import topicwizard.plots.topics as plots
import topicwizard.prepare.topics as prepare
from topicwizard.data import TopicData


def topic_map(
    topic_data: TopicData,
) -> go.Figure:
    """Plots topics on a scatter plot based on the UMAP projections
    of their parameters into 2D space.

    Parameters
    ----------
    topic_data: TopicData
        Inference data from topic modeling.
    """
    x, y = prepare.topic_positions(topic_data["topic_term_matrix"])
    (
        topic_importances,
        _,
        _,
    ) = prepare.topic_importances(
        topic_data["topic_term_matrix"],
        topic_data["document_term_matrix"],
        topic_data["document_topic_matrix"],
    )
    fig = plots.intertopic_map(
        x=x,
        y=y,
        topic_importances=topic_importances,
        topic_names=topic_data["topic_names"],
    )
    return fig


def topic_barcharts(
    topic_data: TopicData,
    top_n: int = 5,
    n_columns: int = 4,
) -> go.Figure:
    """Plots most relevant words as bar charts for every topic.

    Parameters
    ----------
    topic_data: TopicData
        Inference data from topic modeling.
    top_n: int, default 5
        Specifies the number of words to show for each topic.
    n_columns: int, default 4
        Number of columns in the subplot grid.
    """
    vocab = topic_data["vocab"]
    components = topic_data["topic_term_matrix"]
    topic_names = prepare.infer_topic_names(vocab, components)
    n_topics = len(topic_names)
    n_rows = (n_topics // n_columns) + 1
    fig = make_subplots(
        rows=n_rows,
        cols=n_columns,
        subplot_titles=topic_names,
    )
    for topic_id in range(n_topics):
        top_words = prepare.calculate_top_words(
            topic_id=topic_id,
            top_n=top_n,
            components=components,
            vocab=vocab,
        )
        subfig = plots.topic_plot(top_words)
        row, column = (topic_id // n_columns) + 1, (topic_id % n_columns) + 1
        for trace in subfig.data:
            # hiding legend if it isn't the first trace.
            if topic_id:
                trace.showlegend = False
            fig.add_trace(trace, row=row, col=column)
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
    fig.update_xaxes(zerolinecolor="black", zerolinewidth=5)
    fig.update_yaxes(ticks="", showticklabels=False)
    fig.update_xaxes(
        gridcolor="#e5e7eb",
    )
    fig.update_yaxes(
        gridcolor="#e5e7eb",
    )
    return fig


def topic_wordclouds(
    topic_data: TopicData,
    top_n: int = 30,
    n_columns: int = 4,
) -> go.Figure:
    """Plots most relevant words as word clouds for every topic.

    Parameters
    ----------
    topic_data: TopicData
        Inference data from topic modeling.
    top_n: int, default 30
        Specifies the number of words to show for each topic.
    n_columns: int, default 4
        Number of columns in the subplot grid.
    """
    n_topics = topic_data["topic_term_matrix"].shape[0]
    (
        topic_importances,
        term_importances,
        topic_term_importances,
    ) = prepare.topic_importances(
        topic_data["topic_term_matrix"],
        topic_data["document_term_matrix"],
        topic_data["document_topic_matrix"],
    )
    n_rows = (n_topics // n_columns) + 1
    fig = make_subplots(
        rows=n_rows,
        cols=n_columns,
        subplot_titles=topic_data["topic_names"],
        vertical_spacing=0.05,
        horizontal_spacing=0.01,
    )
    for topic_id in range(n_topics):
        top_words = prepare.calculate_top_words(
            topic_id=topic_id,
            top_n=top_n,
            components=topic_term_importances,
            vocab=topic_data["vocab"],
        )
        subfig = plots.wordcloud(top_words)
        row, column = (topic_id // n_columns) + 1, (topic_id % n_columns) + 1
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
