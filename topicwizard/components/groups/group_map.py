from typing import List, Tuple

import numpy as np
from dash_extensions.enrich import DashBlueprint, Input, Output, State, dcc

import topicwizard.plots.groups as plots


def create_group_map(
    group_positions: Tuple[np.ndarray, np.ndarray],
    group_importances: np.ndarray,
    group_names: List[str],
):
    x, y = group_positions

    group_map = DashBlueprint()

    group_map.layout = dcc.Graph(
        id="group_map",
        responsive=True,
        # config=dict(scrollZoom=True),
        # animate=True,
        figure=plots.group_map(x, y, group_importances, group_names),
        className="flex-1",
    )

    group_map.clientside_callback(
        """
        function(currentGroup, currentPlot) {
            const trigerred = window.dash_clientside.callback_context.triggered[0];
            const trigerredId = trigerred.prop_id.split(".")[0];
            const trace = currentPlot.data[0]
            if (currentPlot) {
                const nGroups = trace['x'].length
                const colors = new Array(nGroups).fill('rgb(168 162 158)')
                colors[currentGroup] = '#E03131'
                const marker = {...trace.marker, 'color': colors}
                const newTrace = {...trace, 'marker': marker}
                const newFigure = {...currentPlot, 'data': [newTrace]}
                return newFigure;
            } else {
                return {'data': [], 'layout': {}};
            }
        }
        """,
        Output("group_map", "figure"),
        Input("selected_group", "data"),
        State("group_map", "figure"),
        prevent_initial_call=True,
    )

    group_map.clientside_callback(
        """
        function(clickData, currentValue) {
            if (!clickData) {
                return currentValue;
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
