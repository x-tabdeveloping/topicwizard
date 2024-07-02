from io import BytesIO
from typing import Callable, Dict, List, Optional, Set

import dash_mantine_components as dmc
import joblib
import numpy as np
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    exceptions, html)
from dash_iconify import DashIconify

import topicwizard.blueprints.documents as documents
import topicwizard.blueprints.groups as groups
import topicwizard.blueprints.topics as topics
import topicwizard.blueprints.words as words
from topicwizard.blueprints.template import create_blank_page
from topicwizard.data import TopicData


def create_blueprint(
    vocab: np.ndarray,
    document_term_matrix: np.ndarray,
    document_topic_matrix: np.ndarray,
    topic_term_matrix: np.ndarray,
    document_names: List[str],
    document_representation: np.ndarray,
    corpus: List[str],
    transform: Optional[Callable],
    topic_names: List[str],
    exclude_pages: Set[str],
    group_labels: Optional[List[str]],
    word_positions: Optional[np.ndarray] = None,
    topic_positions: Optional[np.ndarray] = None,
    document_positions: Optional[np.ndarray] = None,
) -> DashBlueprint:
    # --------[ Collecting blueprints ]--------
    topic_blueprint = (
        topics.create_blueprint(
            vocab=vocab,
            document_term_matrix=document_term_matrix,
            document_topic_matrix=document_topic_matrix,
            topic_term_matrix=topic_term_matrix,
            document_names=document_names,
            corpus=corpus,
            topic_names=topic_names,
            topic_positions=topic_positions,
        )
        if "topics" not in exclude_pages
        else create_blank_page("topics")
    )
    documents_blueprint = (
        documents.create_blueprint(
            vocab=vocab,
            document_term_matrix=document_term_matrix,
            document_topic_matrix=document_topic_matrix,
            topic_term_matrix=topic_term_matrix,
            document_names=document_names,
            document_representation=document_representation,
            transform=transform,
            corpus=corpus,
            topic_names=topic_names,
            document_positions=document_positions,
        )
        if "documents" not in exclude_pages
        else create_blank_page("documents")
    )
    words_blueprint = (
        words.create_blueprint(
            vocab=vocab,
            document_term_matrix=document_term_matrix,
            document_topic_matrix=document_topic_matrix,
            topic_term_matrix=topic_term_matrix,
            document_names=document_names,
            corpus=corpus,
            topic_names=topic_names,
            word_positions=word_positions,
        )
        if "words" not in exclude_pages
        else create_blank_page("words")
    )
    groups_blueprint = (
        groups.create_blueprint(
            vocab=vocab,
            document_term_matrix=document_term_matrix,
            document_topic_matrix=document_topic_matrix,
            topic_term_matrix=topic_term_matrix,
            document_names=document_names,
            corpus=corpus,
            topic_names=topic_names,
            group_labels=group_labels,
        )
        if group_labels is not None
        else create_blank_page("groups")
    )
    if group_labels is None:
        exclude_pages = exclude_pages | set(["groups"])
    options = []
    for option in ["Topics", "Words", "Documents", "Groups"]:
        if option.lower() not in exclude_pages:
            options.append(option)
    blueprints = [
        topic_blueprint,
        words_blueprint,
        documents_blueprint,
        groups_blueprint,
    ]

    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()

    app_blueprint.layout = html.Div(
        [
            dcc.Download("download_data"),
            dcc.Store(
                "topic_names",
                data=topic_names,
            ),
            topic_blueprint.layout,
            words_blueprint.layout,
            documents_blueprint.layout,
            groups_blueprint.layout,
            html.Div(
                [
                    dmc.SegmentedControl(
                        id="page_picker",
                        data=options,
                        value=options[0],
                        color="orange",
                        size="md",
                        radius="xl",
                    ),
                    html.Div(className="w-5"),
                    dmc.ActionIcon(
                        DashIconify(
                            icon="material-symbols:cloud-download-outline",
                            width=25,
                        ),
                        id="download_button",
                        size="xl",
                        radius="md",
                        n_clicks=0,
                        color="blue",
                        variant="light",
                    ),
                ],
                className="""
                    flex-row p-3
                    flex-none justify-center flex
                """,
            ),
        ],
        className="""
            fixed w-full h-full flex-col flex items-stretch
            bg-white
        """,
    )

    # --------[ Registering callbacks ]--------
    for blueprint in blueprints:
        blueprint.register_callbacks(app_blueprint)

    @app_blueprint.callback(
        Output("download_data", "data"),
        Input("download_button", "n_clicks"),
        State("topic_names", "data"),
    )
    def download_data(n_clicks: int, topic_names: List[str]):
        if not n_clicks:
            raise exceptions.PreventUpdate
        data = TopicData(
            corpus=corpus,
            vocab=vocab,
            document_term_matrix=document_term_matrix,
            document_topic_matrix=document_topic_matrix,
            topic_term_matrix=topic_term_matrix,
            document_representation=document_representation,
            transform=transform,
            topic_names=topic_names,
            document_positions=document_positions,
            topic_positions=topic_positions,
            word_positions=word_positions,
        )

        def write_joblib(bytes_io: BytesIO):
            joblib.dump(data, filename=bytes_io)

        return dcc.send_bytes(write_joblib, "topic_data.joblib")

    app_blueprint.clientside_callback(
        """
        function(currentPage){
            const visible = 'flex flex-1 flex-col p-3';
            const hidden = 'hidden';
            if (currentPage === 'Topics') {
                return [visible, hidden, hidden, hidden];
            }
            if (currentPage === 'Words') {
                return [hidden, visible, hidden, hidden];
            }
            if (currentPage === 'Documents') {
                return [hidden, hidden, visible, hidden];
            }
            if (currentPage === 'Groups') {
                return [hidden, hidden, hidden, visible];
            }
            return [hidden, hidden, hidden, hidden];
        }
        """,
        Output("topics_container", "className"),
        Output("words_container", "className"),
        Output("documents_container", "className"),
        Output("groups_container", "className"),
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
            if (currentPage === 'Groups') {
                return 'violet';
            }
            return 'black';
        }
        """,
        Output("page_picker", "color"),
        Input("page_picker", "value"),
    )

    return app_blueprint
