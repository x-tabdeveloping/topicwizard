import random
from functools import partial
from typing import List, Optional

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    html)
from sklearn.metrics.pairwise import euclidean_distances

import topicwizard.prepare.documents as prepare
from topicwizard.data import TopicData

from .widget import Widget

mantine_color_sequence_24 = [
    "#fa5252",  # red 6
    "#c92a2a",  # red 9
    "#e64980",  # pink 6
    "#a61e4d",  # pink 9
    "#be4bdb",  # grape 6
    "#862e9c",  # grape 9
    "#7950f2",  # violet 6
    "#5f3dc4",  # violet 9
    "#4c6ef5",  # indigo 6
    "#364fc7",  # indigo 9
    "#228be6",  # blue 6
    "#1864ab",  # blue 9
    "#15aabf",  # cyan 6
    "#0b7285",  # cyan 9
    "#12b886",  # teal 6
    "#087f5b",  # teal 9
    "#40c057",  # green 6
    "#2b8a3e",  # green 9
    "#82c91e",  # lime 6
    "#5c940d",  # lime 9
    "#fab005",  # yellow 6
    "#e67700",  # yellow 9
    "#fd7e14",  # orange 6
    "#d9480f",  # orange 9
]
np.random.default_rng(42).shuffle(mantine_color_sequence_24)


def plot_document_clusters(
    corpus: List[str],
    document_positions: np.ndarray,
    document_topic_matrix: np.ndarray,
    topic_names: List[str],
):
    x, y = document_positions
    document_topic_assignment = np.argmax(document_topic_matrix, axis=1)
    document_topic_assignment = [
        topic_names[i_topic] for i_topic in document_topic_assignment
    ]
    cleaned_text = [" ".join(text.split()) for text in corpus]
    cleaned_text = [text[:150] for text in corpus]
    doc_df = pd.DataFrame(
        dict(
            x=x,
            y=y,
            topic=document_topic_assignment,
            content=cleaned_text,
        )
    )
    fig = px.scatter(
        doc_df,
        x="x",
        y="y",
        color="topic",
        template="plotly_white",
        hover_data={"x": False, "y": False, "topic": True},
        hover_name="content",
        color_discrete_sequence=mantine_color_sequence_24,
    )
    fig.update_xaxes(title="", showticklabels=False)
    fig.update_yaxes(title="", showticklabels=False)
    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),
        legend=dict(
            yanchor="bottom",
            y=0,
            xanchor="right",
            x=1,
            bgcolor="rgba(256, 256, 256, 0.4)",
            bordercolor="black",
            borderwidth=2,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
        ),
        legend_title_text="Topic",
        dragmode="pan",
        font=dict(
            color="black",
            size=16,
        ),
    )
    return fig


def create_document_clusters(
    app_id: str,
    corpus: List[str],
    document_topic_matrix: np.ndarray,
    topic_names: List[str],
    document_representation: np.ndarray,
    document_positions: Optional[np.ndarray] = None,
    **kwargs,
) -> DashBlueprint:
    # --------[ Preparing data ]--------
    if document_positions is None:
        document_positions = prepare.document_positions(document_representation)
    cluster_plot = plot_document_clusters(
        corpus, document_positions, document_topic_matrix, topic_names
    )
    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            dmc.Center(
                dmc.Text(
                    """This widget allows you to explore clusters of documents in your corpus.
                    Zoom in by scrolling and see the first 300 characters in a document by hovering.
                    """,
                    size="sm",
                    fw=400,
                    c="dimmed",
                    className="pb-6",
                ),
            ),
            dcc.Graph(
                figure=cluster_plot,
                className="flex-1 flex",
                config=dict(scrollZoom=True),
            ),
        ],
        className="""
        flex flex-1 flex-col 
        p-3 h-full
        """,
        id=f"document_clusters_{app_id}",
    )

    return app_blueprint


class DocumentClusters(Widget):
    needed_attributes = (
        "corpus",
        "document_topic_matrix",
        "topic_names",
        "document_representation",
    )
    icon = "material-symbols:group-work-outline"
    name = "Document Clusters"
    id_prefix = "document_clusters"

    def create_blueprint(self, topic_data: TopicData, app_id: str = ""):
        return create_document_clusters(app_id, **topic_data)
