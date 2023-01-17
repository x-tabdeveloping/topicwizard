from typing import List, Any
import numpy as np

from dash_extensions.enrich import (
    DashBlueprint,
    html,
    Output,
    Input,
)
import dash_mantine_components as dmc
import topicwizard.blueprints.topics as topics
import topicwizard.blueprints.words as words
import topicwizard.blueprints.documents as documents


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
    # --------[ Collecting blueprints ]--------
    topic_blueprint = topics.create_blueprint(
        vocab=vocab,
        document_term_matrix=document_term_matrix,
        document_topic_matrix=document_topic_matrix,
        topic_term_matrix=topic_term_matrix,
        document_names=document_names,
        corpus=corpus,
        vectorizer=vectorizer,
        topic_model=topic_model,
        topic_names=topic_names,
    )
    documents_blueprint = documents.create_blueprint(
        vocab=vocab,
        document_term_matrix=document_term_matrix,
        document_topic_matrix=document_topic_matrix,
        topic_term_matrix=topic_term_matrix,
        document_names=document_names,
        corpus=corpus,
        vectorizer=vectorizer,
        topic_model=topic_model,
        topic_names=topic_names,
    )
    words_blueprint = words.create_blueprint(
        vocab=vocab,
        document_term_matrix=document_term_matrix,
        document_topic_matrix=document_topic_matrix,
        topic_term_matrix=topic_term_matrix,
        document_names=document_names,
        corpus=corpus,
        vectorizer=vectorizer,
        topic_model=topic_model,
        topic_names=topic_names,
    )
    blueprints = [
        topic_blueprint,
        words_blueprint,
        documents_blueprint,
    ]

    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()

    app_blueprint.layout = html.Div(
        [
            topic_blueprint.layout,
            words_blueprint.layout,
            documents_blueprint.layout,
            html.Div(
                dmc.SegmentedControl(
                    id="page_picker",
                    data=["Topics", "Words", "Documents"],
                    value="Topics",
                    color="orange",
                    size="md",
                    radius="xl",
                ),
                className="""
                    flex-row p-3
                    flex-none justify-center flex
                """,
            ),
        ],
        className="""
            fixed w-full h-full flex-col flex items-stretch
        """,
    )

    # --------[ Registering callbacks ]--------
    for blueprint in blueprints:
        blueprint.register_callbacks(app_blueprint)

    app_blueprint.clientside_callback(
        """
        function(currentPage){
            const visible = 'flex flex-1 flex-col p-3';
            const hidden = 'hidden';
            if (currentPage === 'Topics') {
                return [visible, hidden, hidden];
            }
            if (currentPage === 'Words') {
                return [hidden, visible, hidden];
            }
            if (currentPage === 'Documents') {
                return [hidden, hidden, visible];
            }
            return [hidden, hidden, hidden];
        }
        """,
        Output("topics_container", "className"),
        Output("words_container", "className"),
        Output("documents_container", "className"),
        Input("page_picker", "value"),
    )
    app_blueprint.clientside_callback(
        """
        function(currentPage){
            if (currentPage === 'Topics') {
                return 'orange';
            }
            if (currentPage === 'Words') {
                return 'teal';
            }
            if (currentPage === 'Documents') {
                return 'indigo';
            }
            return 'black';
        }
        """,
        Output("page_picker", "color"),
        Input("page_picker", "value"),
    )

    return app_blueprint
