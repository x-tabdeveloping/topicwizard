from functools import lru_cache
from typing import Tuple, List, Union

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


def create_document_wordcloud(
    document_term_matrix: np.ndarray,
    vocab: np.ndarray,
):
    document_wordcloud = DashBlueprint()

    document_wordcloud.layout = dcc.Graph(
        id="document_wordcloud",
        responsive=True,
        className="flex-1",
    )

    @document_wordcloud.callback(
        Output("document_wordcloud", "figure"),
        Input("selected_document", "data"),
    )
    @lru_cache
    def update_doc_wordcloud(selected_document: Union[int, str]) -> go.Figure:
        if isinstance(selected_document, str):
            selected_document = selected_document.strip()
            selected_document = int(selected_document)
        return plots.document_wordcloud(
            doc_id=selected_document,
            document_term_matrix=document_term_matrix,
            vocab=vocab,
        )

    return document_wordcloud
