"""Window slider component."""

import dash_mantine_components as dmc
from dash_extensions.enrich import DashBlueprint, html


def create_window_slider() -> DashBlueprint:
    window_slider = DashBlueprint()

    window_slider.layout = dmc.Grid(
        [
            dmc.Col(
                html.Div(
                    "WINDOW SIZE:",
                    className="""
                                bg-indigo-50 py-1.5 px-5
                                text-indigo-800 font-medium
                                rounded-xl
                            """,
                ),
                span="content",
            ),
            dmc.Col(
                dmc.Slider(
                    id="window_slider",
                    value=50,
                    min=10,
                    max=500,
                    step=10,
                    size="md",
                    radius="sm",
                    marks=[
                        {"value": 10, "label": "10"},
                        {"value": 50, "label": "50"},
                        {"value": 100, "label": "100"},
                        {"value": 200, "label": "200"},
                        {"value": 500, "label": "500"},
                    ],
                    color="indigo",
                    showLabelOnHover=False,
                ),
                span="auto",
            ),
        ],
    )
    return window_slider
