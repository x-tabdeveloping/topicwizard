from typing import List, Union

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash_extensions.enrich import DashBlueprint, Input, Output, State, dcc, exceptions
from scipy.stats import zscore

import topicwizard.prepare.documents as prepare


def get_highlighted(
    text: str, dominant_topic: int, topic_term_matrix: np.ndarray, vocab: np.ndarray
):
    z_values = zscore(topic_term_matrix[dominant_topic])
    important_terms = list(vocab[z_values > 2.0])
    return dmc.Highlight(text, highlight=important_terms, highlightColor="gray")


def create_document_viewer(
    corpus: List[str],
    vocab: np.ndarray,
    topic_term_matrix: np.ndarray,
    dominant_topic: np.ndarray,
) -> DashBlueprint:
    document_viewer = DashBlueprint()

    document_viewer.layout = dmc.Spoiler(
        id="document_viewer",
        showLabel="Show more",
        hideLabel="Hide",
        maxHeight=100,
        className="px-3 pt-1 pb-8 h-1/5 overflow-y-auto",
    )

    @document_viewer.callback(
        Output("document_viewer", "children"),
        Input("selected_document", "data"),
    )
    def update_content(selected_document: Union[int, str]) -> dmc.Highlight:
        if isinstance(selected_document, str):
            selected_document = selected_document.strip()
            selected_document = int(selected_document)
        display_text = corpus[selected_document][:3000]
        if len(display_text) != len(corpus[selected_document]):
            display_text += "..."
        return get_highlighted(
            display_text,
            dominant_topic[selected_document],
            topic_term_matrix,
            vocab,
        )

    return document_viewer
