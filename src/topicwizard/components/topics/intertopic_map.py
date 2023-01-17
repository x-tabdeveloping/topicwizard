from typing import Tuple, List

import numpy as np

from dash_extensions.enrich import (
    DashBlueprint,
    dcc,
    Input,
    Output,
    State,
)
import topicwizard.plots.topics as plots


def create_intertopic_map(
    topic_positions: Tuple[np.ndarray, np.ndarray],
    topic_importances: np.ndarray,
    topic_names: List[str],
) -> DashBlueprint:
    x, y = topic_positions

    intertopic_map = DashBlueprint()

    intertopic_map.layout = dcc.Graph(
        id="intertopic_map",
        responsive=True,
        config=dict(scrollZoom=True),
        animate=True,
        figure=plots.intertopic_map(
            x=x,
            y=y,
            topic_importances=topic_importances,
            topic_names=topic_names,
        ),
        className="flex-1",
    )

    intertopic_map.clientside_callback(
        """
        function(currentTopic, topicNames, currentPlot) {
            const trigerred = window.dash_clientside.callback_context.triggered[0];
            const trigerredId = trigerred.prop_id.split(".")[0];
            const trace = currentPlot.data[0]
            if (currentPlot && topicNames) {
                const nTopics = topicNames.length
                const colors = new Array(nTopics).fill('rgb(168 162 158)')
                colors[currentTopic] = 'rgb(251 146 60)'
                const marker = {...trace.marker, 'color': colors}
                const newTrace = {...trace, 'marker': marker, 'text': topicNames}
                const newFigure = {...currentPlot, 'data': [newTrace]}
                return newFigure;
            } else {
                return {'data': [], 'layout': {}};
            }
        }
        """,
        Output("intertopic_map", "figure"),
        Input("current_topic", "data"),
        Input("topic_names", "data"),
        State("intertopic_map", "figure"),
        prevent_initial_call=True,
    )
    return intertopic_map
