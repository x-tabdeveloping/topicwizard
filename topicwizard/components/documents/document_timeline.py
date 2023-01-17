from typing import Tuple, List, Union, Any

import numpy as np

from dash_extensions.enrich import (
    DashBlueprint,
    dcc,
    Input,
    Output,
    State,
)
import plotly.graph_objects as go
import topicwizard.plots.documents as plots
import topicwizard.prepare.documents as prepare


def create_timeline(corpus: List[str], vectorizer: Any, topic_model: Any):
    timeline = DashBlueprint()

    timeline.layout = dcc.Graph(
        id="timeline",
        responsive=True,
        className="flex-1",
    )

    @timeline.callback(
        Output("timeline", "figure"),
        Input("selected_document", "data"),
        Input("topic_names", "data"),
        Input("window_slider", "value"),
    )
    def update_timeline(
        selected_document: Union[int, str],
        topic_names: List[str],
        window_size: int,
    ) -> go.Figure:
        if isinstance(selected_document, str):
            selected_document = selected_document.strip()
            selected_document = int(selected_document)
        topic_timeline = prepare.calculate_timeline(
            doc_id=selected_document,
            corpus=corpus,
            vectorizer=vectorizer,
            topic_model=topic_model,
            window_size=window_size,
            step=1,
        )
        return plots.document_timeline(
            topic_timeline=topic_timeline,
            topic_names=topic_names,
        )

    return timeline
