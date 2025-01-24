import random
from functools import partial
from typing import List, Optional

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    html)
from matplotlib.colors import ListedColormap
from sklearn.metrics.pairwise import euclidean_distances

import topicwizard.prepare.words as prepare


def plot_compass(
    topic_x: int,
    topic_y: int,
    topic_term_matrix: np.ndarray,
    vocab: np.ndarray,
    topic_names: List[str],
):
    x = topic_term_matrix[topic_x]
    y = topic_term_matrix[topic_y]
    points = np.array(list(zip(x, y)))
    xx, yy = np.meshgrid(
        np.linspace(np.min(x), np.max(x), 20),
        np.linspace(np.min(y), np.max(y), 20),
    )
    coords = np.array(list(zip(np.ravel(xx), np.ravel(yy))))
    coords = coords + np.random.default_rng(0).normal(
        [0, 0], [0.1, 0.1], size=coords.shape
    )
    dist = euclidean_distances(coords, points)
    idxs = np.argmin(dist, axis=1)
    fig = px.scatter(
        x=x[idxs],
        y=y[idxs],
        text=vocab[idxs],
        template="plotly_white",
    )
    fig = fig.update_traces(
        mode="text", textfont_color="black", marker=dict(color="black")
    ).update_layout(
        xaxis_title=f"{topic_names[topic_x]}",
        yaxis_title=f"{topic_names[topic_y]}",
    )
    fig = fig.update_layout(
        font=dict(family="Times New Roman", color="black", size=21),
        margin=dict(l=5, r=5, t=5, b=5),
        dragmode="pan",
    )
    fig = fig.add_hline(y=0, line_color="black", line_width=4)
    fig = fig.add_vline(x=0, line_color="black", line_width=4)
    return fig


def create_concept_compass(
    vocab: np.ndarray,
    topic_term_matrix: np.ndarray,
    topic_names: List[str],
    **kwargs,
) -> DashBlueprint:
    app_id = random.randint(0, 10_000)
    # --------[ Preparing data ]--------
    default_compass = plot_compass(0, 1, topic_term_matrix, vocab, topic_names)
    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            dmc.Center(
                [
                    dmc.Text(
                        "Concept Compass",
                        size="xl",
                        ta="center",
                        fw=700,
                        className="pb-6",
                    ),
                ]
            ),
            dmc.Group(
                [
                    dmc.Select(
                        label="Select X Semantic Axis",
                        id=f"x_select_{app_id}",
                        value=0,
                        data=[
                            {"value": i_topic, "label": topic_name}
                            for i_topic, topic_name in enumerate(topic_names)
                        ],
                    ),
                    dmc.Select(
                        label="Select Y Semantic Axis",
                        id=f"y_select_{app_id}",
                        value=1,
                        data=[
                            {"value": i_topic, "label": topic_name}
                            for i_topic, topic_name in enumerate(topic_names)
                        ],
                    ),
                ],
                grow=True,
            ),
            dcc.Graph(
                figure=default_compass,
                className="flex-1 flex",
                config=dict(scrollZoom=True),
                id=f"compass_{app_id}",
            ),
        ],
        className="""
        flex flex-1 flex-col 
        p-3 h-full
        """,
        id="word_clusters",
    )

    @app_blueprint.callback(
        Output(f"compass_{app_id}", "figure"),
        Input(f"x_select_{app_id}", "value"),
        Input(f"y_select_{app_id}", "value"),
        prevent_initial_call=True,
    )
    def update_compass(topic_x: int, topic_y: int):
        return plot_compass(topic_x, topic_y, topic_term_matrix, vocab, topic_names)

    return app_blueprint
