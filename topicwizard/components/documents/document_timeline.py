from typing import Any, List, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from dash_extensions.enrich import DashBlueprint, Input, Output, State, dcc

import topicwizard.plots.documents as plots
import topicwizard.prepare.documents as prepare


def create_timeline(
    corpus: List[str], vectorizer: Any, topic_model: Any, topic_colors: np.ndarray
):
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
            topic_colors=topic_colors,
            topic_timeline=topic_timeline,
            topic_names=topic_names,
        )

    return timeline
