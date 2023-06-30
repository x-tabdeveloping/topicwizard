"""External API for creating self-contained figures for words."""
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
import scipy.sparse as spr
from plotly import colors
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
) -> go.Figure:
    """Plots words on a scatter plot based on the UMAP projections
    of their importances in topics into 2D space.
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
    dominant_topic = prepare.dominant_topic(topic_term_matrix)
    tempo = colors.get_colorscale("tempo")
    topic_colors = colors.sample_colorscale(tempo, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    return plots.word_map(x, y, word_frequencies, vocab, dominant_topic, topic_colors)
