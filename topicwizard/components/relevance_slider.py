"""Relevance slider component"""

from dash import html, dcc


relevance_slider = html.Div(
    className="""
        fixed flex flex-none flex-row justify-between items-center
        left-40 bottom-5 h-16 w-96 bg-white shadow
        rounded-full ml-5 px-6 py-6
    """,
    children=[
        html.Div("λ :", className="text-xl text-gray-500"),
        dcc.Slider(
            id="lambda_slider",
            value=1.0,
            min=0.0,
            max=1.0,
            className="flex-1 mt-5",
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ],
)
