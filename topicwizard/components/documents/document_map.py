from typing import Tuple, List

import numpy as np

from dash_extensions.enrich import (
    DashBlueprint,
    dcc,
    Input,
    Output,
    State,
)
import topicwizard.plots.documents as plots


def create_document_map(
    document_positions: Tuple[np.ndarray, np.ndarray],
    dominant_topic: np.ndarray,
    document_names: List[str],
    n_topics: int,
):
    x, y = document_positions

    document_map = DashBlueprint()

    document_map.layout = dcc.Graph(
        id="document_map",
        responsive=True,
        figure=plots.document_map(
            x=x,
            y=y,
            document_names=document_names,
        ),
        className="flex-1",
    )

    document_map.clientside_callback(
        """
        function(selectedDocument, currentPlot) {
            if (!currentPlot){
                return {'data': [], 'layout': {}};
            }
            const trigerred = window.dash_clientside.callback_context.triggered[0];
            const trigerredId = trigerred.prop_id.split(".")[0];
            const trace = currentPlot.data[0];
            const nDocuments = trace.x.length
            const text = new Array(nDocuments).fill('');
            const colors = new Array(nDocuments).fill('#a8a29e');
            if (selectedDocument !== undefined) {
                text[selectedDocument] = trace.customdata[selectedDocument][1];
                colors[selectedDocument] = '#5F3DC4';
            }
            const marker = {...trace.marker, 'color': colors};
            const newTrace = {...trace, 'marker': marker, 'text': text};
            const newFigure = {...currentPlot, 'data': [newTrace]};
            return newFigure;
        }
        """,
        Output("document_map", "figure"),
        Input("selected_document", "data"),
        State("document_map", "figure"),
    )

    return document_map
