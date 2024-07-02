from typing import Any, Callable, List, Optional

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import DashBlueprint, Input, Output, dcc, html
from plotly import colors

import topicwizard.help.documents as help
import topicwizard.prepare.documents as prepare
from topicwizard.components.color_legend import make_color_legend
from topicwizard.components.documents.document_bar import create_document_bar
from topicwizard.components.documents.document_map import create_document_map
from topicwizard.components.documents.document_selector import \
    create_document_selector
from topicwizard.components.documents.document_timeline import create_timeline
from topicwizard.components.documents.document_viewer import \
    create_document_viewer
from topicwizard.components.documents.window_slider import create_window_slider
from topicwizard.help.utils import make_helper


def create_blueprint(
    vocab: np.ndarray,
    document_topic_matrix: np.ndarray,
    topic_term_matrix: np.ndarray,
    document_names: List[str],
    corpus: List[str],
    transform: Optional[Callable],
    document_representation: np.ndarray,
    document_positions: Optional[np.ndarray] = None,
    **kwargs,
) -> DashBlueprint:
    # --------[ Preparing data ]--------
    n_topics = document_topic_matrix.shape[1]
    if document_positions is None:
        document_positions = prepare.document_positions(document_representation)
    dominant_topics = prepare.dominant_topic(
        document_topic_matrix=document_topic_matrix
    )
    # Creating unified color scheme
    color_scheme = colors.get_colorscale("Portland")
    # Factorizing labels
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
        transform=transform,
        topic_colors=topic_colors,
    )
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
                    dmc.Stack(
                        [
                            dmc.Title("Content:", order=3, className="px-3 mt-2"),
                            document_viewer.layout,
                            document_map.layout,
                        ],
                        align="stretch",
                        justify="space-around",
                        className="flex-1",
                    ),
                    dmc.Stack(
                        [
                            document_bar.layout,
                            timeline.layout,
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
                make_helper(
                    dmc.Group(
                        [
                            html.Div(help.DOCUMENT_MAP),
                            html.Div(help.CONTENT),
                            html.Div(help.TIMELINE),
                        ],
                        spacing="lg",
                        grow=1,
                        align="start",
                    ),
                    width="800px",
                ),
                className="fixed bottom-8 right-5",
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
