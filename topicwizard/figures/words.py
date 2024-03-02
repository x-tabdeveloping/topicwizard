"""External API for creating self-contained figures for words."""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import colors
from scipy.stats import zscore

import topicwizard.plots.words as plots
import topicwizard.prepare.words as prepare
from topicwizard.data import TopicData


def word_map(
    topic_data: TopicData,
    z_threshold: float = 2.0,
    topic_axes: Optional[Tuple[Union[str, int], Union[str, int]]] = None,
) -> go.Figure:
    """Plots words on a scatter plot based on UMAP projections
    of their importances in topics into 2D space or by two topic axes.

    Parameters
    ----------
    topic_data: TopicData
        Inference data from topic modeling.
    z_threshold: float, default 2.0
        Z-score frequency threshold over which words get labels on the
        plot. The default roughly corresponds to 95% percentile
        if we assume normal distribution for word frequencies
        (which is probably not the case, see Zipf's law).
        If you find not enough words have labels, lower this number
        if you find there is too much clutter on your graph,
        change this to something higher.
    topic_axes: tuple of str|int, optional
        The topic axes along which the words should be displayed.
        If not specified, the axes on the graph are going to be
        UMAP projections' dimensions.
    """
    topic_names = topic_data["topic_names"]
    if topic_axes is None:
        x, y = prepare.word_positions(topic_data["topic_term_matrix"])
        xaxis_name = "UMAP_1"
        yaxis_name = "UMAP_1"
    else:
        xaxis, yaxis = topic_axes
        if isinstance(xaxis, str):
            xaxis = topic_names.index(xaxis)
        if isinstance(yaxis, str):
            yaxis = topic_names.index(yaxis)
        xaxis_name = topic_names[xaxis]
        yaxis_name = topic_names[yaxis]
        x = topic_data["topic_term_matrix"][xaxis]
        y = topic_data["topic_term_matrix"][yaxis]
    word_frequencies = prepare.word_importances(topic_data["document_term_matrix"])
    freq_z = zscore(word_frequencies)
    dominant_topic = prepare.dominant_topic(topic_data["topic_term_matrix"])
    dominant_topic = np.array(topic_data["topic_names"])[dominant_topic]
    tempo = colors.get_colorscale("tempo")
    n_topics = len(topic_data["topic_names"])
    topic_colors = colors.sample_colorscale(tempo, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    text = np.where(freq_z > z_threshold, topic_data["vocab"], "")
    words_df = pd.DataFrame(
        dict(
            word=topic_data["vocab"],
            text=text,
            dominant_topic=dominant_topic,
            frequency=word_frequencies,
            x=x,
            y=y,
        )
    )
    fig = px.scatter(
        words_df,
        x="x",
        y="y",
        text="text",
        size="frequency",
        color="dominant_topic",
        size_max=100,
        hover_data={
            "word": True,
            "text": False,
            "x": False,
            "y": False,
        },
        template="plotly_white",
    )
    fig.update_xaxes(title=xaxis_name)
    fig.update_yaxes(title=yaxis_name)
    return fig


def word_association_barchart(
    topic_data: TopicData,
    words: Union[List[str], str],
    n_association: int = 0,
    top_n: int = 20,
):
    """Plots bar chart of most important topics for the given words and their closest
    associations in topic space.

    Parameters
    ----------
    topic_data: TopicData
        Inference data from topic modeling.
    words: list[str] or str
        Words you want to start the association from.
    n_association: int, default 0
        Number of words to associate with the given words.
        None get displayed by default.
    top_n: int = 20
        Top N topics to display.
    """
    if isinstance(words, str):
        words = [words]
    word_to_id = {word: id for id, word in enumerate(topic_data["vocab"])}
    try:
        word_ids = [word_to_id[word] for word in words]
    except KeyError as e:
        raise KeyError(
            "One of the provided words is not in the vectorizers vocabulary."
        ) from e
    associated_words = prepare.associated_words(
        word_ids, topic_data["topic_term_matrix"], n_association
    )
    n_topics = topic_data["topic_term_matrix"].shape[0]
    tempo = colors.get_colorscale("Rainbow")
    topic_colors = colors.sample_colorscale(tempo, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    top_topics = prepare.top_topics(
        word_ids,
        associated_words,
        top_n=top_n,
        topic_term_matrix=topic_data["topic_term_matrix"],
        topic_names=topic_data["topic_names"],
    )
    return plots.word_topics_plot(top_topics, topic_colors=topic_colors)
