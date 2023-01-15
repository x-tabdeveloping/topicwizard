from typing import Any, Iterable, Callable

from dash_extensions.enrich import Dash, DashBlueprint
import numpy as np

from topicwizard.prepare.utils import get_vocab, prepare_transformed_data

BlueprintCreator = Callable[..., DashBlueprint]


def prepare_blueprint(
    vectorizer: Any,
    topic_model: Any,
    corpus: Iterable[str],
    create_blueprint: BlueprintCreator,
) -> DashBlueprint:
    vocab = get_vocab(vectorizer)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)
    blueprint = create_blueprint(
        vocab=vocab,
        document_term_matrix=document_term_matrix,
        document_topic_matrix=document_topic_matrix,
        topic_term_matrix=topic_term_matrix,
    )
    return blueprint
