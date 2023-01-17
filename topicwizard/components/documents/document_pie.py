from typing import List, Union

import numpy as np
import plotly.graph_objects as go

from dash_extensions.enrich import (
    DashBlueprint,
    dcc,
    Input,
    Output,
    State,
    exceptions,
)
import topicwizard.plots.documents as plots
import topicwizard.prepare.documents as prepare


def create_document_pie(document_topic_matrix: np.ndarray) -> DashBlueprint:
    topic_importances = prepare.document_topic_importances(
        document_topic_matrix=document_topic_matrix
    )

    document_pie = DashBlueprint()

    document_pie.layout = dcc.Graph(
        id="document_pie",
        responsive=True,
        className="flex-1",
    )

    @document_pie.callback(
        Output("document_pie", "figure"),
        Input("selected_document", "data"),
        Input("topic_names", "data"),
    )
    def update_plot(
        selected_document: Union[int, str], topic_names: List[str]
    ) -> go.Figure:
        if isinstance(selected_document, str):
            selected_document = selected_document.strip()
            selected_document = int(selected_document)
        return plots.document_topic_plot(
            topic_importances=topic_importances[
                topic_importances.doc_id == selected_document
            ],
            topic_names=topic_names,
        )

    return document_pie
