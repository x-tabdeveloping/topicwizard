"""External API for creating self-contained figures for words."""
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import colors
from scipy.stats import zscore
from sklearn.pipeline import Pipeline, make_pipeline

import topicwizard.plots.words as plots
import topicwizard.prepare.words as prepare
from topicwizard.app import split_pipeline
from topicwizard.prepare.topics import infer_topic_names
from topicwizard.prepare.utils import get_vocab, prepare_transformed_data


def word_map(
    corpus: Iterable[str],
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    topic_names: Optional[List[str]] = None,
    z_threshold: float = 2.0,
) -> go.Figure:
    """Plots words on a scatter plot based on the UMAP projections
    of their importances in topics into 2D space.

    Parameters
    ----------
    corpus: iterable of str
        List of all works in the corpus you intend to visualize.
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
    z_threshold: float, default 2.0
        Z-score frequency threshold over which words get labels on the
        plot. The default roughly corresponds to 95% percentile
        if we assume normal distribution for word frequencies
        (which is probably not the case, see Zipf's law).
        If you find not enough words have labels, lower this number
        if you find there is too much clutter on your graph,
        change this to something higher.

    Returns
    -------
    go.Figure
        Map of words.
    """
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    if topic_names is None:
        topic_names = infer_topic_names(pipeline)
    n_topics = len(topic_names)
    vocab = get_vocab(vectorizer)
    corpus = list(corpus)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)
    x, y = prepare.word_positions(topic_term_matrix)
    word_frequencies = prepare.word_importances(document_term_matrix)
    freq_z = zscore(word_frequencies)
    dominant_topic = prepare.dominant_topic(topic_term_matrix)
    dominant_topic = np.array(topic_names)[dominant_topic]
    tempo = colors.get_colorscale("tempo")
    topic_colors = colors.sample_colorscale(tempo, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    text = np.where(freq_z > z_threshold, vocab, "")
    words_df = pd.DataFrame(
        dict(
            word=vocab,
            text=text,
            dominant_topic=dominant_topic,
            frequency=word_frequencies,
            x=x,
            y=y,
        )
    )
    return px.scatter(
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


def word_association_barchart(
    words: Union[List[str], str],
    corpus: Iterable[str],
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    topic_names: Optional[List[str]] = None,
    n_association: int = 0,
    top_n: int = 20,
):
    """Plots bar chart of most important topics for the given words and their closest
    associations in topic space.

    Parameters
    ----------
    corpus: iterable of str
        List of all works in the corpus you intend to visualize.
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
    n_association: int, default 0
        Number of words to associate with the given words.
        None get displayed by default.
    top_n: int = 20
        Top N topics to display.

    Returns
    -------
    go.Figure
        Bar chart of most important topics for the given words and
        their associations.
    """
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    if topic_names is None:
        topic_names = infer_topic_names(pipeline)
    corpus = list(corpus)
    vocab = get_vocab(vectorizer)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)
    if isinstance(words, str):
        words = [words]
    word_to_id = {word: id for id, word in enumerate(vocab)}
    try:
        word_ids = [word_to_id[word] for word in words]
    except KeyError as e:
        raise KeyError(
            "One of the provided words is not in the vectorizers vocabulary."
        ) from e
    associated_words = prepare.associated_words(
        word_ids, topic_term_matrix, n_association
    )
    n_topics = topic_term_matrix.shape[0]
    tempo = colors.get_colorscale("Rainbow")
    topic_colors = colors.sample_colorscale(tempo, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    top_topics = prepare.top_topics(
        word_ids,
        associated_words,
        top_n=top_n,
        topic_term_matrix=topic_term_matrix,
        topic_names=topic_names,
    )
    return plots.word_topics_plot(top_topics, topic_colors=topic_colors)
