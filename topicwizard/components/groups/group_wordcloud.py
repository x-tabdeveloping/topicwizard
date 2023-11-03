from typing import List

import numpy as np
import plotly.graph_objects as go
from dash_extensions.enrich import DashBlueprint, Input, Output, dcc

import topicwizard.plots.groups as plots
import topicwizard.prepare.groups as prepare


def create_group_wordcloud(
    group_term_importances: np.ndarray, vocab: np.ndarray
) -> DashBlueprint:
    group_wordcloud = DashBlueprint()

    group_wordcloud.layout = dcc.Graph(
        id="group_wordcloud",
        responsive=True,
        className="flex-1",
    )

    @group_wordcloud.callback(
        Output("group_wordcloud", "figure"),
        Input("selected_group", "data"),
    )
    def update_plot(selected_group: int) -> go.Figure:
        top_words = prepare.top_words(selected_group, 60, group_term_importances, vocab)
        return plots.wordcloud(top_words)

    return group_wordcloud
