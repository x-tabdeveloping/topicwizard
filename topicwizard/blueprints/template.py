from typing import Any, Iterable, Callable, List, Optional

from dash_extensions.enrich import DashBlueprint

from topicwizard.prepare.utils import get_vocab, prepare_transformed_data

BlueprintCreator = Callable[..., DashBlueprint]


def prepare_blueprint(
    vectorizer: Any,
    topic_model: Any,
    corpus: Iterable[str],
    create_blueprint: BlueprintCreator,
    document_names: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
) -> DashBlueprint:
    corpus = list(corpus)
    n_documents = len(corpus)
    if document_names is None:
        document_names = [f"Document {i}" for i in range(n_documents)]
    vocab = get_vocab(vectorizer)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)
    n_topics = topic_term_matrix.shape[0]
    if topic_names is None:
        topic_names = [f"Topic {i}" for i in range(n_topics)]
    blueprint = create_blueprint(
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
    return blueprint
