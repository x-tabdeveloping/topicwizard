from typing import Any, List

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import DashBlueprint, dcc, html
from plotly import colors
from sklearn.pipeline import Pipeline

import topicwizard.prepare.documents as prepare
from topicwizard.components.color_legend import make_color_legend
from topicwizard.components.documents.document_bar import create_document_bar
from topicwizard.components.documents.document_map import create_document_map
from topicwizard.components.documents.document_selector import create_document_selector
from topicwizard.components.documents.document_timeline import create_timeline
from topicwizard.components.documents.document_viewer import create_document_viewer
from topicwizard.components.documents.document_wordcloud import (
    create_document_wordcloud,
)
from topicwizard.components.documents.window_slider import create_window_slider


def create_blueprint(
    vocab: np.ndarray,
    document_term_matrix: np.ndarray,
    document_topic_matrix: np.ndarray,
    topic_term_matrix: np.ndarray,
    topic_names: List[str],
    document_names: List[str],
    corpus: List[str],
    pipeline: Pipeline,
    **kwargs,
) -> DashBlueprint:
    # --------[ Preparing data ]--------
    n_topics = topic_term_matrix.shape[0]
    document_positions = prepare.document_positions(
        document_topic_matrix=document_topic_matrix
    )
    dominant_topics = prepare.dominant_topic(
        document_topic_matrix=document_topic_matrix
    )
    # Creating unified color scheme
    color_scheme = colors.get_colorscale("Portland")
    topic_colors = colors.sample_colorscale(
        color_scheme, np.arange(n_topics) / n_topics, low=0.25, high=1.0
    )
    topic_colors = np.array(topic_colors)

    # --------[ Collecting blueprints ]--------
    document_map = create_document_map(
        document_names=document_names,
        document_positions=document_positions,
        dominant_topic=dominant_topics,
        n_topics=n_topics,
        topic_colors=topic_colors,
    )
    timeline = create_timeline(
        corpus=corpus,
        transform=pipeline.transform,
        topic_colors=topic_colors,
    )
    # document_wordcloud = create_document_wordcloud(
    #     document_term_matrix=document_term_matrix, vocab=vocab
    # )
    document_bar = create_document_bar(
        document_topic_matrix=document_topic_matrix, topic_colors=topic_colors
    )
    document_selector = create_document_selector(document_names=document_names)
    document_viewer = create_document_viewer(
        corpus=corpus,
        vocab=vocab,
        topic_term_matrix=topic_term_matrix,
        dominant_topic=dominant_topics,
    )
    window_slider = create_window_slider()
    blueprints = [
        document_map,
        document_selector,
        document_bar,
        document_viewer,
        timeline,
        window_slider,
    ]

    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            dcc.Store(
                "selected_document",
                data=0,
            ),
            dmc.Grid(
                [
                    dmc.Col(document_selector.layout, span=5),
                    dmc.Col(window_slider.layout, span=4),
                ],
                columns=10,
                align="center",
                gutter="md",
                justify="space-between",
                mr=5,
            ),
            dmc.Group(
                [
                    document_map.layout,
                    dmc.Stack(
                        [
                            dmc.Title("Content:", order=3, className="px-3 mt-2"),
                            document_viewer.layout,
                            timeline.layout,
                            document_bar.layout,
                        ],
                        align="stretch",
                        justify="space-around",
                        className="flex-1",
                    ),
                ],
                grow=1,
                align="stretch",
                position="apart",
                className="flex-1 p-3",
            ),
            html.Div(
                make_color_legend(topic_names, topic_colors),
                className="""
                bg-white rounded-md px-4 py-2 fixed bottom-5 left-5
                opacity-80 hover:opacity-100 shadow-md
                max-h-96
                scroll-smooth overflow-y-scroll
                """,
            ),
        ],
        className="""
        hidden
        """,
        id="documents_container",
    )

    # --------[ Registering callbacks ]--------
    for blueprint in blueprints:
        blueprint.register_callbacks(app_blueprint)
    return app_blueprint
