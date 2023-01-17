"""Window slider component."""

import dash_mantine_components as dmc
from dash_extensions.enrich import DashBlueprint


def create_window_slider() -> DashBlueprint:
    window_slider = DashBlueprint()

    window_slider.layout = dmc.Grid(
        [
            dmc.Col(
                dmc.Badge(
                    "window size:",
                    size="xl",
                    radius="xl",
                    variant="gradient",
                    gradient={"from": "indigo", "to": "violet", "deg": 105},
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
                    color="violet",
                    showLabelOnHover=False,
                ),
                span="auto",
            ),
        ],
    )
    return window_slider
