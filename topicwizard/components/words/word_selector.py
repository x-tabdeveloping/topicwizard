
import numpy as np

from dash_extensions.enrich import (
    DashBlueprint,
    dcc,
    Input,
    Output,
    State,
)

def create_word_selector(
    vocab: np.ndarray,
):
    terms = [
        {"value": index, "label": term} for index, term in enumerate(vocab)
    ]

    word_selector = DashBlueprint()

    word_selector.layout = dcc.Dropdown(
        id="word_selector",
        placeholder="Select words...",
        value=[],
        options=terms,
        multi=True,
        searchable=True,
        className="min-w-max flex-1",
        clearable=True,
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
