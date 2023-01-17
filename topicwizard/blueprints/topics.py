import functools
from typing import Tuple, List, Any

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash_extensions.enrich import (
    DashBlueprint,
    Input,
    Output,
    State,
    dcc,
    html,
)

import topicwizard.plots.topics as plots
import topicwizard.prepare.topics as prepare
from topicwizard.components.topics.intertopic_map import create_intertopic_map
from topicwizard.components.topics.relevance_slider import relevance_slider
from topicwizard.components.topics.topic_barplot import topic_barplot
from topicwizard.components.topics.topic_namer import topic_namer
from topicwizard.components.topics.topic_switcher import topic_switcher
from topicwizard.components.topics.wordcloud import wordcloud

# ----Clientside Callbacks----
switch_topic = [
    """
    function(nextClicked, prevClicked, clickData, currentTopic) {
        const trigerred = window.dash_clientside.callback_context.triggered[0];
        const trigerredId = trigerred.prop_id.split(".")[0];
        if ((trigerredId === 'next_topic') && (nextClicked !== 0)) {
            return currentTopic + 1;
        }
        if ((trigerredId === 'prev_topic') && (nextClicked !== 0)) {
            return currentTopic - 1;
        }
        if ((trigerredId === 'intertopic_map') && (clickData)) {
            const point = clickData.points[0]
            const topicId = point.customdata[0]
            return topicId;
        }
        return 0;
    }
    """,
    Output("current_topic", "data"),
    Input("next_topic", "n_clicks"),
    Input("prev_topic", "n_clicks"),
    Input("intertopic_map", "clickData"),
    State("current_topic", "data"),
]


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
    topic_positions = prepare.topic_positions(topic_term_matrix)
    (
        topic_importances,
        term_importances,
        topic_term_importances,
    ) = prepare.topic_importances(
        topic_term_matrix, document_term_matrix, document_topic_matrix
    )

    # --------[ Collecting blueprints ]--------
    intertopic_map = create_intertopic_map(
        topic_positions, topic_importances, topic_names
    )
    blueprints = [
        intertopic_map,
        relevance_slider,
        topic_switcher,
        topic_namer,
        topic_barplot,
        wordcloud,
    ]
    # layouts = [blueprint.layout for blueprint in blueprints]

    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            dcc.Store(
                "topic_names",
                data=topic_names,
            ),
            dcc.Store("current_topic", data=0),
            dmc.Grid(
                [
                    dmc.Col(topic_switcher.layout, span="content"),
                    dmc.Col(topic_namer.layout, span=4),
                    dmc.Col(relevance_slider.layout, span=4),
                ],
                columns=10,
                align="center",
                gutter="md",
                justify="space-between",
                mr=5,
            ),
            html.Div(
                [
                    intertopic_map.layout,
                    html.Div(
                        [
                            topic_barplot.layout,
                            wordcloud.layout,
                        ],
                        className="flex-1 flex flex-col items-stretch",
                    ),
                ],
                className="flex-1 flex flex-row items-stretch p-3",
            ),
        ],
        className="""
        flex flex-1 flex-col flex
        p-3
        """,
        id="topics_container",
    )

    # --------[ Registering callbacks ]--------
    @app_blueprint.callback(
        Output("topic_barplot", "figure"),
        Output("wordcloud", "figure"),
        Input("lambda_slider", "value"),
        Input("current_topic", "data"),
    )
    @functools.lru_cache
    def update_plots(
        relevance: float, current_topic: int
    ) -> Tuple[go.Figure, go.Figure]:
        top_bar = prepare.calculate_top_words(
            topic_id=current_topic,
            top_n=30,
            alpha=relevance,
            term_frequency=term_importances,
            topic_term_frequency=topic_term_importances,
            vocab=vocab,
        )
        top_wordcloud = prepare.calculate_top_words(
            topic_id=current_topic,
            top_n=200,
            alpha=relevance,
            term_frequency=term_importances,
            topic_term_frequency=topic_term_importances,
            vocab=vocab,
        )
        bar = plots.topic_plot(top_words=top_bar)
        wordcloud = plots.wordcloud(top_words=top_wordcloud)
        return bar, wordcloud

    app_blueprint.clientside_callback(*switch_topic)
    for blueprint in blueprints:
        blueprint.register_callbacks(app_blueprint)
    return app_blueprint
