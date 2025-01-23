from typing import List

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    html)
from matplotlib.colors import ListedColormap

from topicwizard.plots.topics import wordcloud
from topicwizard.prepare.topics import calculate_top_words


def get_wordclouds(vocab, components):
    wordclouds = []
    colormap = ListedColormap(["#0B1A4A", "#183280", "#005F8F"])
    for i in range(components.shape[0]):
        top_words = calculate_top_words(i, 100, components, vocab)
        wc = wordcloud(top_words, color_scheme=colormap)
        wc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        wordclouds.append(wc)
    return wordclouds


def make_bars(vocab, components):
    bars = []
    for component in components:
        idx = np.argsort(-component)[:10]
        top_words = pd.DataFrame(dict(word=vocab[idx], importance=component[idx]))
        top_words["text"] = (
            "<b>"
            + top_words["word"]
            + " ("
            + top_words["importance"].map(lambda x: f"{x:.2f}")
            + ")</b>"
        )
        top_words = top_words.sort_values("importance", ascending=True)
        barplot = px.bar(
            top_words,
            y="word",
            x="importance",
            text="text",
            orientation="h",
            template="plotly_white",
        )
        barplot.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            xaxis_title="",
            yaxis_title="",
            font=dict(color="#0B1A4A"),
        )
        barplot.update_traces(
            marker_color="white", marker_line_color="#0B1A4A", marker_line_width=1.5
        )
        barplot.update_xaxes(showticklabels=False)
        barplot.update_yaxes(showticklabels=False)
        bars.append(barplot)
    return bars


def make_table(topic_document_imp, corpus):
    idx = np.argsort(-topic_document_imp)[:10]
    head = html.Thead(
        html.Tr(
            [
                html.Th(dmc.Text("Most Relevant Documents", className="pl-1 pb-1")),
                html.Th(dmc.Center(dmc.Text("Relevance")), className="w-1/6 pb-1"),
            ]
        )
    )
    rows = []
    for i_doc in idx:
        doc = corpus[i_doc]
        imp = topic_document_imp[i_doc]
        imp = f"{imp:.2f}"
        row = html.Tr(
            [
                html.Th(
                    dmc.Spoiler(
                        showLabel="...",
                        hideLabel="Hide",
                        maxHeight=50,
                        className="overflow-y-auto pl-1",
                        children=[dmc.Text(doc, fw=100, size="xs")],
                        style={"max-height": "180px"},
                    )
                ),
                html.Th(dmc.Center(dmc.Text(imp, fw=600, size="xs"))),
            ],
        )
        rows.append(row)
    body = html.Tbody(rows)
    return dmc.Table(
        [head, body],
        verticalSpacing=2.5,
        horizontalSpacing=2.5,
    )


def create_document_tables(corpus, document_topic_matrix) -> List[dmc.Table]:
    tables = []
    for topic_document_imp in document_topic_matrix.T:
        tables.append(make_table(topic_document_imp, corpus))
    return tables


def create_topic_browser(
    vocab: np.ndarray,
    corpus: List[str],
    document_topic_matrix: np.ndarray,
    topic_term_matrix: np.ndarray,
    topic_names: List[str],
    **kwargs,
) -> DashBlueprint:
    # --------[ Preparing data ]--------
    top_words = []
    for component in topic_term_matrix:
        idx = np.argsort(-component)[:10]
        top_words.append(list(vocab[idx]))
    document_tables = create_document_tables(corpus, document_topic_matrix)
    wordclouds = get_wordclouds(vocab, topic_term_matrix)
    bars = make_bars(vocab, topic_term_matrix)
    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            dmc.Center(
                [
                    dmc.Text(
                        "Topic Browser",
                        size="xl",
                        ta="center",
                        fw=700,
                        className="pb-6",
                    ),
                ]
            ),
            dmc.Accordion(
                chevronPosition="right",
                variant="separated",
                children=[
                    dmc.AccordionItem(
                        [
                            dmc.AccordionControl(
                                html.Div(
                                    [
                                        dmc.Text(topic_names[i_topic]),
                                        dmc.Text(
                                            "Keywords: "
                                            + ", ".join(top_words[i_topic]),
                                            size="sm",
                                            fw=400,
                                            c="dimmed",
                                        ),
                                    ]
                                ),
                            ),
                            dmc.AccordionPanel(
                                [
                                    dmc.Group(
                                        [
                                            dcc.Graph(
                                                figure=bars[i_topic],
                                                style={"height": "200x"},
                                                className="w-2/3",
                                            ),
                                            dcc.Graph(
                                                figure=wordclouds[i_topic],
                                                className="w-1/3",
                                            ),
                                        ],
                                        grow=True,
                                    ),
                                    dmc.Spoiler(
                                        children=document_tables[i_topic],
                                        maxHeight=250,
                                        showLabel="Expand Table",
                                        hideLabel="Collapse Table",
                                    ),
                                ]
                            ),
                        ],
                        value=topic_names[i_topic],
                    )
                    for i_topic in range(len(topic_names))
                ],
            ),
        ],
        className="""
        flex flex-1 flex-col flex
        p-3
        """,
        id="topic_browser",
    )
    return app_blueprint
