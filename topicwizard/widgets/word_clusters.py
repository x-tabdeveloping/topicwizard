from typing import List, Optional

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    html)

import topicwizard.prepare.words as prepare
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


def produce_map(
    topic_term_matrix: np.ndarray,
    vocab: np.ndarray,
    word_positions: np.ndarray,
    topic_names: List[str],
):
    word_topic_assignment = np.argmax(topic_term_matrix, axis=0)
    word_topic_assignment = [topic_names[i_topic] for i_topic in word_topic_assignment]
    word_importance = np.max(topic_term_matrix, axis=0)
    top_min = np.min(word_importance[np.argsort(-word_importance)[:150]])
    show_text = np.where(word_importance > top_min, vocab, "")
    word_df = pd.DataFrame(
        dict(
            x=word_positions[0],
            y=word_positions[1],
            topic=word_topic_assignment,
            show_text=show_text,
            word=vocab,
            importance=word_importance,
        )
    )
    fig = px.scatter(
        word_df,
        x="x",
        y="y",
        text="show_text",
        color="topic",
        size="importance",
        template="plotly_white",
        category_orders={"topic": topic_names},
        hover_name="word",
        hover_data={
            "importance": True,
            "show_text": False,
            "topic": True,
            "x": False,
            "y": False,
        },
        color_discrete_sequence=mantine_color_sequence_24,
        size_max=60,
    )
    text_size = (np.sqrt(word_importance) / np.sqrt(np.max(word_importance))) * 22 + 8
    fig.update_traces(textfont=dict(size=text_size))
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
    fig.update_xaxes(title="", showticklabels=False)
    fig.update_yaxes(title="", showticklabels=False)
    return fig


def create_word_clusters(
    app_id: str,
    vocab: np.ndarray,
    corpus: List[str],
    topic_term_matrix: np.ndarray,
    topic_names: List[str],
    word_positions: Optional[np.ndarray] = None,
    **kwargs,
) -> DashBlueprint:
    # --------[ Preparing data ]--------
    if word_positions is None:
        word_positions = prepare.word_positions(topic_term_matrix=topic_term_matrix)
    word_map = produce_map(topic_term_matrix, vocab, word_positions, topic_names)
    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            dmc.Center(
                dmc.Text(
                    """ This widget allows you to explore topics as clusters of concepts
                    on an interactive map.
                    Zoom by scrolling and hover over individual words to gain more information about them.
                    Larger points represent words more important to their respective topics.
                    """,
                    size="sm",
                    fw=400,
                    c="dimmed",
                    className="pb-6",
                ),
            ),
            dcc.Graph(
                figure=word_map, className="flex-1 flex", config=dict(scrollZoom=True)
            ),
        ],
        className="""
        flex flex-1 flex-col
        p-3 h-full
        """,
        id=f"word_clusters_{app_id}",
    )
    return app_blueprint


class ConceptClusters(Widget):
    needed_attributes = (
        "vocab",
        "corpus",
        "topic_term_matrix",
        "topic_names",
    )
    icon = "material-symbols:category-outline"
    name = "Concept Clusters"
    id_prefix = "word_clusters"

    def __init__(self):
        super().__init__()

    def create_blueprint(self, topic_data: TopicData, app_id: str = ""):
        return create_word_clusters(app_id, **topic_data)
