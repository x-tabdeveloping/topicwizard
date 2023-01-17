from typing import List
import functools

import numpy as np
import plotly.graph_objects as go

from dash_extensions.enrich import (
    DashBlueprint,
    dcc,
    Input,
    Output,
    State,
    exceptions,
)
import topicwizard.plots.words as plots
import topicwizard.prepare.words as prepare


def list_to_tuple(function):
    """Nasty decorator hack to stop lru_cache from
    complaining that list is not hashable.
    """

    def wrapper(*args):
        args = [tuple(x) if type(x) == list else x for x in args]
        result = function(*args)
        result = tuple(result) if type(result) == list else result
        return result

    return wrapper


def create_word_barplot(topic_term_matrix: np.ndarray) -> DashBlueprint:
    word_barplot = DashBlueprint()

    word_barplot.layout = dcc.Graph(
        id="word_barplot",
        responsive=True,
        animate=True,
        className="flex-1",
    )

    @word_barplot.callback(
        Output("word_barplot", "figure"),
        Input("selected_words", "data"),
        Input("associated_words", "data"),
        Input("topic_names", "data"),
        prevent_initial_call=True,
    )
    @list_to_tuple
    @functools.lru_cache
    def update_barplot(
        selected_words: List[int],
        associated_words: List[int],
        topic_names: List[str],
    ) -> go.Figure:
        if not selected_words or not topic_names:
            raise exceptions.PreventUpdate
        top_topics = prepare.top_topics(
            selected_words=selected_words,
            associated_words=associated_words,
            top_n=30,
            topic_term_matrix=topic_term_matrix,
            topic_names=topic_names,
        )
        return plots.word_topics_plot(top_topics)

    return word_barplot
