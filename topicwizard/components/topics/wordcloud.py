import functools

import plotly.graph_objects as go
from dash_extensions.enrich import DashBlueprint, Input, Output, dcc

import topicwizard.plots.topics as plots
import topicwizard.prepare.topics as prepare


def create_wordcloud(topic_term_matrix, vocab):
    wordcloud = DashBlueprint()
    top_bar = prepare.calculate_top_words(
        topic_id=0,
        top_n=200,
        components=topic_term_matrix,
        vocab=vocab,
    )
    fig = plots.wordcloud(top_bar)
    wordcloud.layout = dcc.Graph(
        id="wordcloud", responsive=True, animate=False, className="flex-1", figure=fig
    )

    @wordcloud.callback(
        Output("wordcloud", "figure"),
        Input("current_topic", "data"),
    )
    @functools.lru_cache
    def update(current_topic: int) -> go.Figure:
        top_bar = prepare.calculate_top_words(
            topic_id=current_topic,
            top_n=200,
            components=topic_term_matrix,
            vocab=vocab,
        )
        return plots.wordcloud(top_bar)

    return wordcloud
