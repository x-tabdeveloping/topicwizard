"""External API for creating self-contained figures for documents."""
from typing import Any, Iterable, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as spr
from plotly import colors
from sklearn.pipeline import Pipeline, make_pipeline

import topicwizard.plots.documents as plots
import topicwizard.prepare.documents as prepare
from topicwizard.app import split_pipeline
from topicwizard.prepare.topics import infer_topic_names
from topicwizard.prepare.utils import get_vocab, prepare_transformed_data


def document_map(
    corpus: Iterable[str],
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    topic_names: Optional[List[str]] = None,
    document_names: Optional[List[str]] = None,
    representation: Literal["term", "topic"] = "term",
) -> go.Figure:
    """Plots documents on a scatter plot based on the UMAP projections
    of their representations in the model into 2D space.

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
    representation: {"term", "topic"}, default "term"
        Determines which representation of the documents should be
        projected to 2D space and displayed.
        If 'term', representations returned from the vectorizer
        will be used, if 'topic', representations returned by
        the topic model will be used. This can be particularly
        advantageous with non-bag-of-words topic models.

    Returns
    -------
    go.Figure
        Map of documents.
    """
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    if topic_names is None:
        topic_names = infer_topic_names(pipeline)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)
    n_docs = document_term_matrix.shape[0]
    if document_names is None:
        document_names = [f"Document {i}" for i in range(n_docs)]
    if representation == "term":
        x, y = prepare.document_positions(document_term_matrix)
    else:
        x, y = prepare.document_positions(document_topic_matrix)
    dominant_topic = prepare.dominant_topic(document_topic_matrix)
    dominant_topic = np.array(topic_names)[dominant_topic]
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
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    topic_names: Optional[List[str]] = None,
    top_n: int = 8,
) -> go.Figure:
    """Plots distribution of topics in the given documents on a bar chart.

    Parameters
    ----------
    documents: str or list of str
        A single document or list of documents.
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

    Returns
    -------
    go.Figure
        Bar chart of topic distribution.
    """
    if isinstance(documents, str):
        documents = [documents]
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    if topic_names is None:
        topic_names = infer_topic_names(pipeline)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, documents)
    topic_importances = prepare.document_topic_importances(document_topic_matrix)
    topic_importances = topic_importances.groupby(["topic_id"]).sum().reset_index()
    n_topics = document_topic_matrix.shape[-1]
    twilight = colors.get_colorscale("Portland")
    topic_colors = colors.sample_colorscale(twilight, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    return plots.document_topic_barplot(
        topic_importances, topic_names, topic_colors, top_n=top_n
    )


def document_topic_timeline(
    document: str,
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    topic_names: Optional[List[str]] = None,
    window_size: int = 10,
    step: int = 1,
) -> go.Figure:
    """Plots timeline of topics inside a single document.

    Parameters
    ----------
    document: str
        A single document.
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
    window_size: int, default 10
        Windows of tokens to take for timeline construction.
    step: int, default 1
        Step size of the window in number of tokens.

    Returns
    -------
    go.Figure
        Line chart of topic timeline in the document.
    """
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    if topic_names is None:
        topic_names = infer_topic_names(pipeline)
    timeline = prepare.calculate_timeline(
        doc_id=0,
        corpus=[document],
        transform=pipeline.transform,
        window_size=window_size,
        step=step,
    )
    n_topics = len(topic_names)
    twilight = colors.get_colorscale("Portland")
    topic_colors = colors.sample_colorscale(twilight, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    return plots.document_timeline(timeline, topic_names, topic_colors)
