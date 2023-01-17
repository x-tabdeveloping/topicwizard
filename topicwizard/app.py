from typing import Any, Iterable, List, Optional
import webbrowser

from dash_extensions.enrich import Dash, DashBlueprint
from sklearn.pipeline import Pipeline

from topicwizard.blueprints.template import prepare_blueprint
from topicwizard.blueprints.app import create_blueprint


def get_app_blueprint(
    vectorizer: Any,
    topic_model: Any,
    corpus: Iterable[str],
    document_names: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
) -> DashBlueprint:
    blueprint = prepare_blueprint(
        vectorizer=vectorizer,
        topic_model=topic_model,
        corpus=corpus,
        document_names=document_names,
        topic_names=topic_names,
        create_blueprint=create_blueprint,
    )
    return blueprint


def get_dash_app(
    vectorizer: Any,
    topic_model: Any,
    corpus: Iterable[str],
    document_names: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
) -> Dash:
    """Returns topicwizard Dash application.

    Parameters
    ----------
    vectorizer: Vectorizer
        Sklearn compatible vectorizer, that turns texts into
        bag-of-words representations.
    topic_model: TopicModel
        Sklearn compatible topic model, that can transform documents
        into topic distributions.
    corpus: iterable of str
        List of all works in the corpus you intend to visualize.
    document_names: Optional[List[str]] = None,
        List of document names in the corpus, if not provided documents will
        be labeled 'Document <index>'.
    topic_names: Optional[List[str]] = None,
        List of topic names in the corpus, if not provided topics will initially
        be labeled 'Topic <index>'.

    Returns
    -------
    Dash
        Dash application object for topicwizard.
    """
    blueprint = get_app_blueprint(
        vectorizer=vectorizer,
        topic_model=topic_model,
        corpus=corpus,
        document_names=document_names,
        topic_names=topic_names,
    )
    app = Dash(
        blueprint=blueprint,
        title="Topic visualization",
        external_scripts=[
            {
                "src": "https://cdn.tailwindcss.com",
            },
            {
                "src": "https://kit.fontawesome.com/9640e5cd85.js",
                "crossorigin": "anonymous",
            },
        ],
    )
    return app


def visualize(
    corpus: Iterable[str],
    vectorizer: Optional[Any] = None,
    topic_model: Optional[Any] = None,
    pipeline: Optional[Pipeline] = None,
    document_names: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
    port: int = 8050,
) -> None:
    """Visualizes your topic model with topicwizard.

    Parameters
    ----------
    corpus: iterable of str
        List of all works in the corpus you intend to visualize.
    vectorizer: Vectorizer, default None
        Sklearn compatible vectorizer, that turns texts into
        bag-of-words representations.
    topic_model: TopicModel, default None
        Sklearn compatible topic model, that can transform documents
        into topic distributions.
    pipeline: Optional[Pipeline] = None,
        Sklearn compatible pipeline, that has two components:
        a vectorizer and a topic model.
        Ignored if vectorizer and topic_model are provided.
    document_names: Optional[List[str]] = None,
        List of document names in the corpus, if not provided documents will
        be labeled 'Document <index>'.
    topic_names: Optional[List[str]] = None,
        List of topic names in the corpus, if not provided topics will initially
        be labeled 'Topic <index>'.
    port: int
        Port where the application should run in localhost. Defaults to 8050.
    """
    if (vectorizer is None) and (topic_model is None):
        assert (
            pipeline is not None
        ), "Either pipeline or vectorizer and topic model have to be provided"
        (vectorizer, _), (topic_model, _) = pipeline.steps
    app = get_dash_app(
        vectorizer=vectorizer,
        topic_model=topic_model,
        corpus=corpus,
        document_names=document_names,
        topic_names=topic_names,
    )
    webbrowser.open(f"localhost:{port}", new=0, autoraise=True)
    app.run_server(port=port, debug=False)
