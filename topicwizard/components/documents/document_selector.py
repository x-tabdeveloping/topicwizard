from typing import List

from dash_extensions.enrich import (
    DashBlueprint,
    Input,
    Output,
    dcc,
    State,
)


def create_document_selector(
    document_names: List[str],
) -> DashBlueprint:
    docs = [
        {"value": index, "label": name}
        for index, name in enumerate(document_names)
    ]

    document_selector = DashBlueprint()

    document_selector.layout = dcc.Dropdown(
        id="document_selector",
        placeholder="Select document...",
        options=docs,
        searchable=True,
        className="min-w-max flex-1",
    )

    document_selector.clientside_callback(
        """
        function(selectorValue) {
            return selectorValue;
        }
        """,
        Output("selected_document", "data"),
        Input("document_selector", "value"),
    )

    document_selector.clientside_callback(
        """
        function(clickData, currentValue) {
            if (!clickData) {
                return currentValue;
            }
            const point = clickData.points[0]
            const documentId = point.customdata[0]
            return documentId;
        }
        """,
        Output("document_selector", "value"),
        Input("document_map", "clickData"),
        State("document_selector", "value"),
    )
    return document_selector
