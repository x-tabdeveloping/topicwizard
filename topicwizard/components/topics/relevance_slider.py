"""Relevance slider component"""

import dash_mantine_components as dmc
from dash_extensions.enrich import DashBlueprint, html

relevance_slider = DashBlueprint()

relevance_slider.layout = dmc.Grid(
    [
        dmc.Col(
            html.Div(
                "LAMBDA:",
                className="""
                        bg-amber-50 py-1.5 px-5
                        text-amber-800 font-medium
                        rounded-xl
                    """,
            ),
            span="content",
        ),
        dmc.Col(
            dmc.Slider(
                id="lambda_slider",
                value=1.0,
                min=0.0,
                max=1.0,
                step=0.1,
                size="md",
                radius="sm",
                marks=[
                    {"value": value / 5, "label": f"{value*20}%"}
                    for value in range(5 + 1)
                ],
                color="orange",
                showLabelOnHover=False,
            ),
            span="auto",
        ),
    ],
)
