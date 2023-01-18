from typing import List, Any
import numpy as np
from dash_extensions.enrich import (
    DashBlueprint,
    dcc,
    html,
)
import dash_mantine_components as dmc

import topicwizard.prepare.words as prepare
from topicwizard.components.words.word_map import create_word_map
from topicwizard.components.words.word_selector import create_word_selector
from topicwizard.components.words.word_barplot import create_word_barplot
from topicwizard.components.words.association_slider import (
    create_association_slider,
)


def create_blueprint(
    vocab: np.ndarray,
    document_term_matrix: np.ndarray,
    document_topic_matrix: np.ndarray,
    topic_term_matrix: np.ndarray,
    document_names: List[str],
    corpus: List[str],
    vectorizer: Any,
    topic_model: Any,
    topic_names: List[str],
) -> DashBlueprint:
    # --------[ Preparing data ]--------
    word_positions = prepare.word_positions(
        topic_term_matrix=topic_term_matrix
    )
    word_frequencies = prepare.word_importances(document_term_matrix)

    # --------[ Collecting blueprints ]--------
    word_map = create_word_map(
        word_positions=word_positions,
        word_frequencies=word_frequencies,
        vocab=vocab,
    )
    word_selector = create_word_selector(vocab=vocab)
    association_slider = create_association_slider(
        topic_term_matrix=topic_term_matrix,
    )
    word_barplot = create_word_barplot(topic_term_matrix=topic_term_matrix)
    blueprints = [word_map, word_selector, word_barplot, association_slider]

    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            dcc.Store("selected_words", data=[]),
            dcc.Store("associated_words", data=[]),
            dcc.Store("vocab", data=vocab.tolist()),
            dmc.Grid(
                [
                    dmc.Col(
                        word_selector.layout,
                        span="content",
                    ),
                    dmc.Col(association_slider.layout, span=4),
                ],
                columns=10,
                align="center",
                gutter="md",
                justify="space-between",
                mr=5,
            ),
            html.Div(
                [
                    word_map.layout,
                    word_barplot.layout,
                ],
                className="flex-1 flex flex-row items-stretch p-3",
            ),
        ],
        className="""
        hidden
        """,
        id="words_container",
    )

    # --------[ Registering callbacks ]--------
    for blueprint in blueprints:
        blueprint.register_callbacks(app_blueprint)
    return app_blueprint
