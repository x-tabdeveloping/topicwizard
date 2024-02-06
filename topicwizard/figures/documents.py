"""External API for creating self-contained figures for documents."""
from typing import Any, Iterable, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import colors
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

import topicwizard.plots.documents as plots
import topicwizard.prepare.documents as prepare
from topicwizard.prepare.data import prepare_topic_data
from topicwizard.prepare.topics import infer_topic_names


def document_map(
    corpus: Iterable[str],
    model: Union[Pipeline, TransformerMixin],
    topic_names: Optional[List[str]] = None,
    document_names: Optional[List[str]] = None,
    document_representations: Optional[np.ndarray] = None,
) -> go.Figure:
    """Plots documents on a scatter plot based on the UMAP projections
    of their representations in the model into 2D space.

    Parameters
    ----------
    corpus: iterable of str
        List of all works in the corpus you intend to visualize.
    model: Pipeline or TransformerMixin
        Bow topic pipeline or contextual topic model.
    topic_names: list of str, default None
        List of topic names in the corpus, if not provided
        topic names will be inferred.
    document_names: list of str, default None
        Names of documents to be displayed.
    document_representations: ndarray of shape (n_docs, n_dims), default None
        Document representations to project into 2D space.
        If not specified, either BoW or contextual representations
        will be used depending on the model.

    Returns
    -------
    go.Figure
        Map of documents.
    """
    topic_data = prepare_topic_data(
        corpus=corpus,
        model=model,
        topic_names=topic_names,
        document_names=document_names,
        document_representations=document_representations,
    )
    x, y = prepare.document_positions(topic_data["document_representation"])
    dominant_topic = prepare.dominant_topic(topic_data["document_topic_matrix"])
    dominant_topic = np.array(topic_data["topic_names"])[dominant_topic]
    words_df = pd.DataFrame(
        dict(
            dominant_topic=dominant_topic,
            x=x,
            y=y,
            document_name=document_names,
        )
    )
    return px.scatter(
        words_df,
        x="x",
        y="y",
        color="dominant_topic",
        hover_data={
            "dominant_topic": True,
            "document_name": True,
            "x": False,
            "y": False,
        },
        template="plotly_white",
    )


def document_topic_distribution(
    documents: Union[List[str], str],
    model: Union[Pipeline, TransformerMixin],
    topic_names: Optional[List[str]] = None,
    top_n: int = 8,
) -> go.Figure:
    """Plots distribution of topics in the given documents on a bar chart.

    Parameters
    ----------
    documents: str or list of str
        A single document or list of documents.
    model: Pipeline or TransformerMixin
        Bow topic pipeline or contextual topic model.
    topic_names: list of str, default None
        List of topic names in the corpus, if not provided
        topic names will be inferred.
    document_names: list of str, default None
        Names of documents to be displayed.
    top_n: int, default 8
        Number of topics to display.

    Returns
    -------
    go.Figure
        Bar chart of topic distribution.
    """
    if isinstance(documents, str):
        documents = [documents]
    topic_data = prepare_topic_data(
        corpus=documents,
        model=model,
        topic_names=topic_names,
    )
    topic_importances = prepare.document_topic_importances(
        topic_data["document_topic_matrix"]
    )
    topic_importances = topic_importances.groupby(["topic_id"]).sum().reset_index()
    n_topics = topic_data["document_topic_matrix"].shape[-1]
    twilight = colors.get_colorscale("Portland")
    topic_colors = colors.sample_colorscale(twilight, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    return plots.document_topic_barplot(
        topic_importances, topic_data["topic_names"], topic_colors, top_n=top_n
    )


def document_topic_timeline(
    document: str,
    model: Union[Pipeline, TransformerMixin],
    topic_names: Optional[List[str]] = None,
    window_size: int = 10,
    step: int = 1,
) -> go.Figure:
    """Plots timeline of topics inside a single document.

    Parameters
    ----------
    document: str
        A single document.
    model: Pipeline or TransformerMixin
        Bow topic pipeline or contextual topic model.
    topic_names: list of str, default None
        List of topic names in the corpus, if not provided
        topic names will be inferred.
    window_size: int, default 10
        Windows of tokens to take for timeline construction.
    step: int, default 1
        Step size of the window in number of tokens.

    Returns
    -------
    go.Figure
        Line chart of topic timeline in the document.
    """
    try:
        transform = model.transform
    except AttributeError:
        raise ValueError(
            "Looks like your model is transductive, "
            "you can only generate timelines with inductive models."
        )
    if isinstance(model, Pipeline):
        _, vectorizer = model.steps[0]
        _, topic_model = model.steps[-1]
        components = topic_model.components_
        vocab = vectorizer.get_feature_names_out()
    else:
        components = model.components_
        vocab = model.get_vocab()
    if topic_names is None:
        topic_names = infer_topic_names(vocab, components)
    timeline = prepare.calculate_timeline(
        doc_id=0,
        corpus=[document],
        transform=transform,
        window_size=window_size,
        step=step,
    )
    n_topics = len(topic_names)
    twilight = colors.get_colorscale("Portland")
    topic_colors = colors.sample_colorscale(twilight, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    return plots.document_timeline(timeline, topic_names, topic_colors)
