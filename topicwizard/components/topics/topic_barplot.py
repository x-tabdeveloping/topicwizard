import functools

import plotly.graph_objects as go
from dash_extensions.enrich import DashBlueprint, Input, Output, dcc

import topicwizard.plots.topics as plots
import topicwizard.prepare.topics as prepare


def create_topic_barplot(topic_term_matrix, vocab):
    topic_barplot = DashBlueprint()
    top_bar = prepare.calculate_top_words(
        topic_id=0,
        top_n=30,
        components=topic_term_matrix,
        vocab=vocab,
    )
    fig = plots.topic_plot(top_bar)
    topic_barplot.layout = dcc.Graph(
        id="topic_barplot",
        responsive=True,
        animate=False,
        className="flex-1",
        figure=fig,
    )

    @topic_barplot.callback(
        Output("topic_barplot", "figure"),
        Input("current_topic", "data"),
    )
    @functools.lru_cache
    def update(current_topic: int) -> go.Figure:
        top_bar = prepare.calculate_top_words(
            topic_id=current_topic,
            top_n=30,
            components=topic_term_matrix,
            vocab=vocab,
        )
        return plots.topic_plot(top_bar)

    return topic_barplot
