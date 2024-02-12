import os
import subprocess
import sys
import threading
import time
import warnings
from typing import Callable, Iterable, List, Literal, Optional, Set, Union

import joblib
import numpy as np
from dash_extensions.enrich import Dash
from sklearn.pipeline import Pipeline

from topicwizard.blueprints.app import create_blueprint
from topicwizard.data import TopicData
from topicwizard.model_interface import TopicModel
from topicwizard.pipeline import TopicPipeline


def is_notebook() -> bool:
    return "ipykernel" in sys.modules


def is_colab() -> bool:
    return "google.colab" in sys.modules


PageName = Literal["topics", "documents", "words"]


def get_dash_app(
    topic_data: TopicData,
    exclude_pages: Optional[Set[PageName]] = None,
    document_names: Optional[List[str]] = None,
    group_labels: Optional[List[str]] = None,
) -> Dash:
    """Returns topicwizard Dash application.

    Parameters
    ----------
    topic_data: TopicData
        Data about topical inference.
    exclude_pages: set of {"topics", "documents", "words"}, optional
        Pages to exclude from the app.
    document_names: list of str, default None
        List of document names in the corpus, if not provided documents will
        be labeled 'Document <index>'.
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
    if exclude_pages is None:
        exclude_pages = set()
    blueprint = create_blueprint(
        **topic_data,
        document_names=document_names
        or [f"Document {i}" for i, _ in enumerate(topic_data["corpus"])],
        group_labels=group_labels,
        exclude_pages=exclude_pages,
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


def filter_nan_docs(topic_data: TopicData) -> None:
    """Filters out documents, the topical content of which contains nans.
    NOTE: The function works in place.
    """
    nan_documents = np.isnan(topic_data["document_topic_matrix"]).any(axis=1)
    n_nan_docs = np.sum(nan_documents)
    if n_nan_docs:
        warnings.warn(
            f"{n_nan_docs} documents had nan values in the output of the topic model,"
            " these are removed in preprocessing and will not be visible in the app."
        )
        topic_data["corpus"] = list(np.array(topic_data["corpus"])[~nan_documents])
        topic_data["document_topic_matrix"] = topic_data["document_topic_matrix"][
            ~nan_documents
        ]
        topic_data["document_term_matrix"] = topic_data["document_term_matrix"][
            ~nan_documents
        ]


def visualize(
    corpus: Optional[List[str]] = None,
    model: Optional[Union[Pipeline, TopicModel]] = None,
    topic_data: Optional[TopicData] = None,
    document_names: Optional[List[str]] = None,
    exclude_pages: Optional[Iterable[PageName]] = None,
    group_labels: Optional[List[str]] = None,
    port: int = 8050,
    **kwargs,
) -> Optional[threading.Thread]:
    """Visualizes your topic model with topicwizard.

    Parameters
    ----------
    corpus: list[str], optional
        List of all works in the corpus you intend to visualize.
    model: Pipeline or TopicModel, optional
        Bag of words topic pipeline or contextual topic model.
    topic_data: TopicData, optional
        Data about topical inference over a corpus.
    document_names: list of str, default None
        List of document names in the corpus, if not provided documents will
        be labeled 'Document <index>'.
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
    print("Preprocessing")
    if isinstance(model, Pipeline):
        model = TopicPipeline.from_pipeline(model)
    if topic_data is None:
        if (model is None) or (corpus is None):
            raise TypeError(
                "Either corpus and model or topic_data has to be specified."
            )
        topic_data = model.prepare_topic_data(corpus, **kwargs)
    exclude_pages = set() if exclude_pages is None else set(exclude_pages)
    # We filter out all documents that contain nans
    nan_documents = np.isnan(topic_data["document_topic_matrix"]).any(axis=1)
    n_nan_docs = np.sum(nan_documents)
    if n_nan_docs:
        warnings.warn(
            f"{n_nan_docs} documents had nan values in the output of the topic model,"
            " these are removed in preprocessing and will not be visible in the app."
        )
        topic_data["corpus"] = list(np.array(topic_data["corpus"])[~nan_documents])
        topic_data["document_topic_matrix"] = topic_data["document_topic_matrix"][
            ~nan_documents
        ]
        topic_data["document_term_matrix"] = topic_data["document_term_matrix"][
            ~nan_documents
        ]
        document_names = list(np.array(document_names)[~nan_documents])
        group_labels = list(np.array(group_labels)[~nan_documents])
    app = get_dash_app(
        topic_data=topic_data,
        document_names=document_names,
        exclude_pages=exclude_pages,
        group_labels=group_labels,
    )
    return run_app(app, port=port)
