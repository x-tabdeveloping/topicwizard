from string import Template
from typing import List, Tuple

import numpy as np
from dash_extensions.enrich import DashBlueprint, Input, Output, State, dcc

import topicwizard.plots.groups as plots


def create_group_map(
    group_positions: Tuple[np.ndarray, np.ndarray],
    group_importances: np.ndarray,
    group_names: List[str],
    dominant_topic: np.ndarray,
    topic_colors: np.ndarray,
):
    x, y = group_positions

    group_map = DashBlueprint()

    group_map.layout = dcc.Graph(
        id="group_map",
        responsive=True,
        # config=dict(scrollZoom=True),
        # animate=True,
        figure=plots.group_map(
            x, y, group_importances, group_names, dominant_topic, topic_colors
        ),
        className="flex-1",
    )
    n_groups = len(set(group_names))

    group_map.clientside_callback(
        Template(
            """
        function(currentGroup, currentPlot) {
            const trigerred = window.dash_clientside.callback_context.triggered[0];
            const trigerredId = trigerred.prop_id.split(".")[0];
            const trace = currentPlot.data[0]
            if (currentPlot) {
                const textSize = new Array($n_groups).fill(10);
                textSize[currentGroup] = 22
                const textFont = {'size': textSize}
                const nGroups = trace['x'].length
                const newTrace = {...trace, 'textfont': textFont}
                const newFigure = {...currentPlot, 'data': [newTrace]}
                return newFigure;
            } else {
                return {'data': [], 'layout': {}};
            }
        }
        """
        ).substitute(n_groups=n_groups),
        Output("group_map", "figure"),
        Input("selected_group", "data"),
        State("group_map", "figure"),
        prevent_initial_call=True,
    )

    group_map.clientside_callback(
        """
        function(clickData, currentValue) {
            if (!clickData) {
                return 0;
            }
            const point = clickData.points[0]
            const groupId = point.customdata[0]
            return groupId;
        }
        """,
        Output("selected_group", "data"),
        Input("group_map", "clickData"),
        State("selected_group", "data"),
    )
    return group_map
