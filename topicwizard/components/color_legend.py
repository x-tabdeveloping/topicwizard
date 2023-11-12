from typing import List

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import html


def make_color_circle(color: str):
    return html.Div(
        style={"background-color": color},
        className="w-5 h-5 rounded-full border-black border-4",
    )


def make_color_entry(name: str, color: str):
    return dmc.Group([make_color_circle(color), dmc.Text(name, size="sm")])


def make_color_legend(names: List[str], colors: np.ndarray):
    return dmc.Stack(
        [make_color_entry(name, color) for name, color in zip(names, colors)]
    )
