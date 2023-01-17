from typing import List, Any

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import (
    DashBlueprint,
    dcc,
    html,
    html,
)

import topicwizard.prepare.documents as prepare
from topicwizard.components.documents.document_map import create_document_map
from topicwizard.components.documents.window_slider import create_window_slider
from topicwizard.components.documents.document_pie import create_document_pie
from topicwizard.components.documents.document_timeline import create_timeline
from topicwizard.components.documents.document_selector import (
    create_document_selector,
)
from topicwizard.components.documents.document_wordcloud import (
    create_document_wordcloud,
)


def create_blueprint(
    vocab: np.ndarray,
    document_term_matrix: np.ndarray,
    document_topic_matrix: np.ndarray,
    topic_term_matrix: np.ndarray,
    document_names: List[str],
    corpus: List[str],
    vectorizer: Any,
    topic_model: Any,
    topic_names: List[str],
) -> DashBlueprint:

    # --------[ Preparing data ]--------
    n_topics = topic_term_matrix.shape[0]
    document_positions = prepare.document_positions(
        document_term_matrix=document_term_matrix
    )
    dominant_topics = prepare.dominant_topic(
        document_topic_matrix=document_topic_matrix
    )

    # --------[ Collecting blueprints ]--------
    document_map = create_document_map(
        document_names=document_names,
        document_positions=document_positions,
        dominant_topic=dominant_topics,
        n_topics=n_topics,
    )
    timeline = create_timeline(
        corpus=corpus, vectorizer=vectorizer, topic_model=topic_model
    )
    document_wordcloud = create_document_wordcloud(
        document_term_matrix=document_term_matrix, vocab=vocab
    )
    document_selector = create_document_selector(document_names=document_names)
    window_slider = create_window_slider()
    document_pie = create_document_pie(
        document_topic_matrix=document_topic_matrix
    )
    blueprints = [
        document_map,
        document_selector,
        document_wordcloud,
        document_pie,
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
                    dmc.Col(document_selector.layout, span="content"),
                    dmc.Col(window_slider.layout, span=4),
                ],
                columns=10,
                align="center",
                gutter="md",
                justify="space-between",
                mr=5,
            ),
            html.Div(
                [
                    document_map.layout,
                    html.Div(
                        [
                            timeline.layout,
                            html.Div(
                                [
                                    document_pie.layout,
                                    document_wordcloud.layout,
                                ],
                                className="flex-1 flex flex-row items-stretch",
                            ),
                        ],
                        className="flex-1 flex flex-col items-stretch",
                    ),
                ],
                className="flex-1 flex flex-row items-stretch p-3",
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
