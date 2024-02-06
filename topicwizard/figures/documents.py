"""External API for creating self-contained figures for documents."""
from typing import List, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import colors

import topicwizard.plots.documents as plots
import topicwizard.prepare.documents as prepare
from topicwizard.data import TopicData


def document_map(topic_data: TopicData, document_metadata: pd.DataFrame) -> go.Figure:
    x, y = prepare.document_positions(topic_data["document_representation"])
    dominant_topic = prepare.dominant_topic(topic_data["document_topic_matrix"])
    dominant_topic = np.array(topic_data["topic_names"])[dominant_topic]
    docs_df = document_metadata.copy()
    docs_df = docs_df.assign(
        dominant_topic=dominant_topic,
        x=x,
        y=y,
    )
    hover_data = {
        "dominant_topic": True,
        "x": False,
        "y": False,
    }
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
    if isinstance(documents, str):
        documents = [documents]
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
    topic_data: TopicData, document: str, window_size: int = 10, step_size: int = 1
) -> go.Figure:
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
