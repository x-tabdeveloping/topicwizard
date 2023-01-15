import numpy as np

from dash_extensions.enrich import (
    DashBlueprint,
    dcc,
    html,
    Output,
    Input,
    State,
)
import dash_mantine_components as dmc
from topicwizard.blueprints.template import prepare_blueprint
import topicwizard.blueprints.topics as topics
import topicwizard.blueprints.words as words


def create_blueprint(
    vocab: np.ndarray,
    document_term_matrix: np.ndarray,
    document_topic_matrix: np.ndarray,
    topic_term_matrix: np.ndarray,
) -> DashBlueprint:
    # --------[ Collecting blueprints ]--------
    topic_blueprint = topics.create_blueprint(
        vocab, document_term_matrix, document_topic_matrix, topic_term_matrix
    )
    words_blueprint = words.create_blueprint(
        vocab, document_term_matrix, document_topic_matrix, topic_term_matrix
    )
    blueprints = [
        topic_blueprint,
        words_blueprint,
    ]

    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()

    app_blueprint.layout = html.Div(
        [
            topic_blueprint.layout,
            words_blueprint.layout,
            html.Div(
                dmc.SegmentedControl(
                    id="page_picker",
                    data=["Topics", "Words"],
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
                return [visible, hidden];
            }
            if (currentPage === 'Words') {
                return [hidden, visible];
            }
            return [hidden, hidden];
        }
        """,
        Output("topics_container", "className"),
        Output("words_container", "className"),
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
            return 'black';
        }
        """,
        Output("page_picker", "color"),
        Input("page_picker", "value"),
    )

    return app_blueprint
