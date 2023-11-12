import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, List, Literal, Optional, Set

import joblib
from dash_extensions.enrich import Dash, DashBlueprint
from sklearn.pipeline import Pipeline

from topicwizard.blueprints.app import create_blueprint
from topicwizard.blueprints.template import prepare_blueprint


def is_notebook() -> bool:
    return "ipykernel" in sys.modules


def is_colab() -> bool:
    return "google.colab" in sys.modules


def get_app_blueprint(
    pipeline: Pipeline,
    corpus: Iterable[str],
    document_names: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
    *args,
    **kwargs,
) -> DashBlueprint:
    blueprint = prepare_blueprint(
        pipeline=pipeline,
        corpus=corpus,
        document_names=document_names,
        topic_names=topic_names,
        create_blueprint=create_blueprint,
        *args,
        **kwargs,
    )
    return blueprint


PageName = Literal["topics", "documents", "words"]


def get_dash_app(
    corpus: Iterable[str],
    exclude_pages: Set[PageName],
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    document_names: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
    group_labels: Optional[List[str]] = None,
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
    document_names: list of str, default None
        List of document names in the corpus, if not provided documents will
        be labeled 'Document <index>'.
    topic_names: list of str, default None
        List of topic names in the corpus, if not provided, topic
        labels will be inferred.
    group_labels: list of str or None, default None
        List of preexisting labels for the documents.
        You can pass it along if you have genre labels for example.
        In this case an additional page will get created with information
        about how these groups relate to topics and words in the corpus.

    Returns
    -------
    Dash
        Dash application object for topicwizard.
    """
    if pipeline is None:
        pipeline = Pipeline([("Vectorizer", vectorizer), ("Model", topic_model)])
    blueprint = get_app_blueprint(
        pipeline=pipeline,
        corpus=corpus,
        document_names=document_names,
        topic_names=topic_names,
        exclude_pages=exclude_pages,
        group_labels=group_labels,
    )
    app = Dash(
        __name__,
        blueprint=blueprint,
        title="topicwizard",
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


def load_app(filename: str, exclude_pages: Optional[Iterable[PageName]] = None) -> Dash:
    """Loads and prepares saved app from disk.

    Parameters
    ----------
    filename: str
        Path to the file where the data is stored.

    Returns
    -------
    Dash
        Dash application.
    """
    data = joblib.load(filename)
    if exclude_pages is None:
        exclude_pages = set()
    else:
        exclude_pages = set(exclude_pages)
    return get_dash_app(**data, exclude_pages=exclude_pages)


def open_url(url: str) -> None:
    if sys.platform == "win32":
        os.startfile(url)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", url])
    else:
        try:
            subprocess.Popen(["xdg-open", url])
        except OSError:
            print("Please open a browser on: " + url)


def run_silent(app: Dash, port: int) -> Callable:
    def _run_silent():
        import logging
        import warnings

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app.run_server(port=port)

    return _run_silent


def run_app(
    app: Dash,
    port: int = 8050,
) -> Optional[threading.Thread]:
    url = f"http://127.0.0.1:{port}/"
    if is_colab():
        from google.colab import output  # type: ignore

        thread = threading.Thread(target=run_silent(app, port))
        thread.start()
        time.sleep(4)

        print("Open in browser:")
        output.serve_kernel_port_as_window(
            port, anchor_text="Click this link to open topicwizard."
        )
        return thread

    # elif is_notebook():
    #     from IPython.display import IFrame, display
    #
    #     thread = threading.Thread(target=run_silent(app, port))
    #     thread.start()
    #     time.sleep(4)
    #     display(IFrame(src=url, width="1200", height="1000"))
    #     return thread
    else:
        open_url(url)
        app.run_server(port=port)


def load(
    filename: str,
    exclude_pages: Optional[Iterable[PageName]] = None,
    port: int = 8050,
) -> Optional[threading.Thread]:
    """Visualizes topic model data loaded from disk.

    Parameters
    ----------
    filename: str
        Path to the file where the data is stored.
    exclude_pages: iterable of {"topics", "documents", "words"}
        Set of pages you want to exclude from the application.
        This can be relevant as with larger corpora for example,
        calculating UMAP embeddings for documents or words can take
        a long time and you might not be interested in them.
    port: int, default 8050
        Port where the application should run in localhost. Defaults to 8050.

    Returns
    -------
    Thread or None
        Returns a Thread if running in a Jupyter notebook (so you can close the server)
        returns None otherwise.
    """
    print("Preparing data")
    exclude_pages = set() if exclude_pages is None else set(exclude_pages)
    app = load_app(filename, exclude_pages=exclude_pages)
    return run_app(app, port=port)


def split_pipeline(
    vectorizer: Any, topic_model: Any, pipeline: Optional[Pipeline]
) -> tuple[Any, Any]:
    """Check which arguments are provided,
    raises error if the arguments are not satisfactory, and if needed
    splits Pipeline into vectorizer and topic model."""
    if (vectorizer is None) or (topic_model is None):
        if pipeline is None:
            raise TypeError(
                "Either pipeline, or vectorizer and topic model have to be provided"
            )
        _, vectorizer = pipeline.steps[0]
        _, topic_model = pipeline.steps[-1]
    return vectorizer, topic_model


def visualize(
    corpus: Iterable[str],
    vectorizer: Optional[Any] = None,
    topic_model: Optional[Any] = None,
    pipeline: Optional[Pipeline] = None,
    document_names: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
    exclude_pages: Optional[Iterable[PageName]] = None,
    group_labels: Optional[List[str]] = None,
    port: int = 8050,
) -> Optional[threading.Thread]:
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
    pipeline: Pipeline, default None
        Sklearn compatible pipeline, that has two components:
        a vectorizer and a topic model.
        Ignored if vectorizer and topic_model are provided.
    document_names: list of str, default None
        List of document names in the corpus, if not provided documents will
        be labeled 'Document <index>'.
    topic_names: list of str, default None
        List of topic names in the corpus, if not provided topic
        names will be inferred.
    exclude_pages: iterable of {"topics", "documents", "words"}
        Set of pages you want to exclude from the application.
        This can be relevant as with larger corpora for example,
        calculating UMAP embeddings for documents or words can take
        a long time and you might not be interested in them.
    port: int, default 8050
        Port where the application should run in localhost. Defaults to 8050.
    group_labels: list of str or None, default None
        List of preexisting labels for the documents.
        You can pass it along if you have genre labels for example.
        In this case an additional page will get created with information
        about how these groups relate to topics and words in the corpus.


    Returns
    -------
    Thread or None
        Returns a Thread if running in a Jupyter notebook (so you can close the server)
        returns None otherwise.
    """
    if pipeline is None:
        pipeline = Pipeline([("Vectorizer", vectorizer), ("Model", topic_model)])
    exclude_pages = set() if exclude_pages is None else set(exclude_pages)
    print("Preprocessing")
    if topic_names is None and hasattr(pipeline, "topic_names"):
        topic_names = pipeline.topic_names  # type: ignore
    app = get_dash_app(
        pipeline=pipeline,
        corpus=corpus,
        document_names=document_names,
        topic_names=topic_names,
        exclude_pages=exclude_pages,
        group_labels=group_labels,
    )
    return run_app(app, port=port)
