from typing import List, Optional

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    html)

from topicwizard.plots.topics import wordcloud
from topicwizard.prepare.topics import calculate_top_words


def get_wordclouds(vocab, components):
    wordclouds = []
    for i in range(components.shape[0]):
        top_words = calculate_top_words(i, 100, components, vocab)
        wc = wordcloud(top_words, color_scheme="cividis_r")
        wc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        wordclouds.append(wc)
    return wordclouds


def make_table(topic_document_imp, corpus):
    idx = np.argsort(-topic_document_imp)[:10]
    head = html.Thead(
        html.Tr(
            [
                html.Th("Most Relevant Documents"),
                html.Th("Document Relevance", className="w-1/6"),
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
                        className="overflow-y-auto",
                        children=[dmc.Text(doc, fw=100)],
                        style={"max-height": "400px"},
                    )
                ),
                html.Th(dmc.Center(dmc.Text(imp, fw=600))),
            ],
        )
        rows.append(row)
    body = html.Tbody(rows)
    return dmc.Table([head, body])


def create_document_tables(corpus, document_topic_matrix) -> List[dmc.Table]:
    tables = []
    for topic_document_imp in document_topic_matrix.T:
        tables.append(make_table(topic_document_imp, corpus))
    return tables


def create_topic_browser(
    vocab: np.ndarray,
    corpus: List[str],
    document_term_matrix: np.ndarray,
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
                        className="pb-8",
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
                                    dmc.Grid(
                                        [
                                            dmc.Col(
                                                dmc.Spoiler(
                                                    children=document_tables[i_topic],
                                                    maxHeight=500,
                                                    showLabel="Expand Table",
                                                    hideLabel="Collapse Table",
                                                ),
                                                span=6,
                                            ),
                                            dmc.Col(
                                                dcc.Graph(
                                                    figure=wordclouds[i_topic],
                                                    style={"height": "500px"},
                                                ),
                                                span=1,
                                            ),
                                        ],
                                        grow=True,
                                    )
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
