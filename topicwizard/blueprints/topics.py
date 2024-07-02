from typing import List, Optional

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    html)

import topicwizard.help.topics as help
import topicwizard.plots.topics as plots
import topicwizard.prepare.topics as prepare
from topicwizard.components.topics.intertopic_map import create_intertopic_map
from topicwizard.components.topics.topic_barplot import create_topic_barplot
from topicwizard.components.topics.topic_namer import topic_namer
from topicwizard.components.topics.topic_switcher import topic_switcher
from topicwizard.components.topics.wordcloud import create_wordcloud
from topicwizard.help.utils import make_helper

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
    topic_names: List[str],
    topic_positions: Optional[np.ndarray] = None,
    **kwargs,
) -> DashBlueprint:
    # --------[ Preparing data ]--------
    if topic_positions is None:
        topic_positions = prepare.topic_positions(topic_term_matrix)
    topic_importances = document_topic_matrix.sum(axis=0)
    # --------[ Collecting blueprints ]--------
    intertopic_map = create_intertopic_map(
        topic_positions, topic_importances, topic_names
    )
    topic_barplot = create_topic_barplot(topic_term_matrix, vocab)
    wordcloud = create_wordcloud(topic_term_matrix, vocab)
    blueprints = [
        intertopic_map,
        topic_switcher,
        topic_namer,
        topic_barplot,
        wordcloud,
    ]
    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            dcc.Store("current_topic", data=0),
            dmc.Grid(
                [
                    dmc.Col(topic_switcher.layout, span="content"),
                    dmc.Col(topic_namer.layout, span=6),
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
                    topic_barplot.layout,
                    wordcloud.layout,
                ],
                className="flex-1 flex flex-row items-stretch p-3",
            ),
            html.Div(
                make_helper(
                    dmc.Group(
                        [
                            html.Div(help.TOPIC_MAP),
                            html.Div(help.TOPIC_WORDS),
                        ],
                        spacing="lg",
                        grow=1,
                        align="start",
                    ),
                    width="800px",
                ),
                className="fixed bottom-8 left-5",
            ),
        ],
        className="""
        flex flex-1 flex-col flex
        p-3
        """,
        id="topics_container",
    )
    app_blueprint.clientside_callback(*switch_topic)
    for blueprint in blueprints:
        blueprint.register_callbacks(app_blueprint)
    return app_blueprint
