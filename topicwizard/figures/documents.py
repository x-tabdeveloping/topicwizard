"""External API for creating self-contained figures for documents."""
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import colors

import topicwizard.plots.documents as plots
import topicwizard.prepare.documents as prepare
from topicwizard.data import TopicData


def document_map(
    topic_data: TopicData, document_metadata: Optional[pd.DataFrame] = None
) -> go.Figure:
    """Projects documents into 2d space and displays them on a scatter plot.

    Parameters
    ----------
    topic_data: TopicData
        Inference data from topic modeling.
    document_metadata: DataFrame, optional
        Metadata you want displayed when hovering over documents on the graph.
    """
    x, y = prepare.document_positions(topic_data["document_representation"])
    dominant_topic = prepare.dominant_topic(topic_data["document_topic_matrix"])
    dominant_topic = np.array(topic_data["topic_names"])[dominant_topic]
    display_data = dict(
        dominant_topic=dominant_topic,
        x=x,
        y=y,
    )
    if document_metadata is not None:
        docs_df = document_metadata.copy()
        docs_df = docs_df.assign(**display_data)
    else:
        docs_df = pd.DataFrame(display_data)
    hover_data = {
        "dominant_topic": True,
        "x": False,
        "y": False,
    }
    if document_metadata is not None:
        for column in document_metadata.columns:
            hover_data[column] = True
    return px.scatter(
        docs_df,
        x="x",
        y="y",
        color="dominant_topic",
        hover_data=hover_data,
        template="plotly_white",
    )


def document_topic_distribution(
    topic_data: TopicData, documents: Union[List[str], str], top_n: int = 8
) -> go.Figure:
    """Displays topic distribution on a bar plot for a document
    or a set of documents.

    Parameters
    ----------
    topic_data: TopicData
        Inference data from topic modeling.
    documents: list[str] or str
        Documents to display topic distribution for.
    top_n: int, default 8
        Number of topics to display at most.
    """
    transform = topic_data["transform"]
    if transform is None:
        raise TypeError(
            "Topic model doesn't have a transform method, and is possibly transductive."
        )
    if isinstance(documents, str):
        documents = [documents]
    topic_importances = prepare.document_topic_importances(transform(documents))
    topic_importances = topic_importances.groupby(["topic_id"]).sum().reset_index()
    n_topics = topic_data["document_topic_matrix"].shape[-1]
    twilight = colors.get_colorscale("Portland")
    topic_colors = colors.sample_colorscale(twilight, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    return plots.document_topic_barplot(
        topic_importances, topic_data["topic_names"], topic_colors, top_n=top_n
    )


def document_topic_timeline(
    topic_data: TopicData, document: str, window_size: int = 10, step_size: int = 1
) -> go.Figure:
    """Projects documents into 2d space and displays them on a scatter plot.

    Parameters
    ----------
    topic_data: TopicData
        Inference data from topic modeling.
    document: str
        Document to display the timeline for.
    window_size: int, default 10
        The windows over which topic inference should be run.
    step_size: int, default 1
        Size of the steps for the rolling window.
    """
    timeline = prepare.calculate_timeline(
        doc_id=0,
        corpus=[document],
        transform=topic_data["transform"],  # type: ignore
        window_size=window_size,
        step=step_size,
    )
    topic_names = topic_data["topic_names"]
    n_topics = len(topic_names)
    twilight = colors.get_colorscale("Portland")
    topic_colors = colors.sample_colorscale(twilight, np.arange(n_topics) / n_topics)
    topic_colors = np.array(topic_colors)
    return plots.document_timeline(timeline, topic_names, topic_colors)
