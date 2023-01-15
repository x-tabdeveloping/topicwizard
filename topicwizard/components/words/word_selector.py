from typing import Tuple

import numpy as np

from dash_extensions.enrich import (
    DashBlueprint,
    dcc,
    Input,
    Output,
    State,
)
import dash_mantine_components as dmc
import topicwizard.plots.words as plots


def create_word_selector(
    vocab: np.ndarray,
):
    terms = [
        {"value": index, "label": term} for index, term in enumerate(vocab)
    ]

    word_selector = DashBlueprint()

    word_selector.layout = dmc.MultiSelect(
        id="word_selector",
        label="",
        placeholder="Select words...",
        value=[],
        data=terms,
        searchable=True,
        clearable=True,
        nothingFound="No options found",
    )

    word_selector.clientside_callback(
        """
        function(selectorValue) {
            return selectorValue;
        }
        """,
        Output("selected_words", "data"),
        Input("word_selector", "value"),
    )

    word_selector.clientside_callback(
        """
        function(clickData, currentValue) {
            if (!clickData) {
                return currentValue;
            }
            const point = clickData.points[0]
            const wordId = point.customdata[0]
            return currentValue.concat([wordId]);
        }
        """,
        Output("word_selector", "value"),
        Input("word_map", "clickData"),
        State("word_selector", "value"),
    )
    return word_selector
