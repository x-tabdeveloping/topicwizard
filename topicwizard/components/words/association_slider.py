"""Association slider component"""
from typing import List

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import DashBlueprint, Input, Output, html

import topicwizard.prepare.words as prepare


def create_association_slider(topic_term_matrix: np.ndarray) -> DashBlueprint:
    association_slider = DashBlueprint()

    association_slider.layout = dmc.Grid(
        [
            dmc.Col(
                html.Div(
                    "ASSOCIATIONS:",
                    className="""
                        bg-teal-50 py-1.5 px-5
                        text-teal-800 font-medium
                        rounded-xl
                    """,
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
                    color="teal",
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
    def update_associated_words(selected_words: List[int], n_associations: int):
        selected_words = [int(word) for word in selected_words]
        if not selected_words or not n_associations:
            return []
        return prepare.associated_words(
            selected_words=selected_words,
            topic_term_matrix=topic_term_matrix,
            n_association=n_associations,
        )

    return association_slider
