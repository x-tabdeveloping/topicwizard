"""Association slider component"""
from typing import List

import numpy as np
import dash_mantine_components as dmc
from dash_extensions.enrich import DashBlueprint, Output, Input

import topicwizard.prepare.words as prepare


def create_association_slider(topic_term_matrix: np.ndarray) -> DashBlueprint:
    association_slider = DashBlueprint()

    association_slider.layout = dmc.Grid(
        [
            dmc.Col(
                dmc.Badge(
                    "associations:",
                    size="xl",
                    radius="xl",
                    variant="gradient",
                    gradient={"from": "teal", "to": "cyan", "deg": 105},
                ),
                span="content",
            ),
            dmc.Col(
                dmc.Slider(
                    id="association_slider",
                    value=5,
                    min=0,
                    max=40,
                    step=1,
                    size="md",
                    radius="sm",
                    marks=[
                        {"value": value * 5, "label": f"{value*5}"}
                        for value in range(9)
                    ],
                    color="cyan",
                    showLabelOnHover=False,
                ),
                span="auto",
            ),
        ],
    )

    @association_slider.callback(
        Output("associated_words", "data"),
        Input("selected_words", "data"),
        Input("association_slider", "value"),
    )
    def update_associated_words(
        selected_words: List[int], n_associations: int
    ):
        if not selected_words or not n_associations:
            return []
        return prepare.associated_words(
            selected_words=selected_words,
            topic_term_matrix=topic_term_matrix,
            n_association=n_associations,
        )

    return association_slider
