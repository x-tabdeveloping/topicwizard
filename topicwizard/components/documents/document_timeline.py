from typing import Callable, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from dash_extensions.enrich import DashBlueprint, Input, Output, State, dcc

import topicwizard.plots.documents as plots
import topicwizard.prepare.documents as prepare
from topicwizard.plots.utils import text_plot


def create_timeline(
    corpus: List[str],
    topic_colors: np.ndarray,
    transform: Optional[Callable[[List[str]], np.ndarray]] = None,
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
        if transform is None:
            return text_plot(
                "This topic model cannot dynamically infer topical content."
            )
        if isinstance(selected_document, str):
            selected_document = selected_document.strip()
            selected_document = int(selected_document)
        try:
            topic_timeline = prepare.calculate_timeline(
                doc_id=selected_document,
                corpus=corpus,
                transform=transform,
                window_size=window_size,
                step=1,
            )
        except ValueError:
            return text_plot(
                "This document is not long enough to be broken into windows."
            )
        return plots.document_timeline(
            topic_colors=topic_colors,
            topic_timeline=topic_timeline,
            topic_names=topic_names,
        )

    return timeline
