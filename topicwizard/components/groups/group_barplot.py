from typing import List

import numpy as np
import plotly.graph_objects as go
from dash_extensions.enrich import DashBlueprint, Input, Output, dcc

import topicwizard.plots.groups as plots
import topicwizard.prepare.groups as prepare


def create_group_barplot(
    group_topic_importances: np.ndarray, topic_colors: np.ndarray
) -> DashBlueprint:
    group_barplot = DashBlueprint()

    group_barplot.layout = dcc.Graph(
        id="group_barplot",
        responsive=True,
        animate=True,
        className="flex-1",
    )

    @group_barplot.callback(
        Output("group_barplot", "figure"),
        Input("selected_group", "data"),
        Input("topic_names", "data"),
    )
    def update_plot(selected_group: int, topic_names: List[str]) -> go.Figure:
        top_topics = prepare.top_topics(
            selected_group, 10, group_topic_importances, topic_names
        )
        return plots.group_topics_barchart(top_topics, topic_colors)

    return group_barplot
