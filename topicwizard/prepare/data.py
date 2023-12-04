from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TypedDict,
)
from warnings import warn

import numpy as np
from dash_extensions.enrich import DashBlueprint, html
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from topicwizard.pipeline import split_pipeline
from topicwizard.prepare.topics import infer_topic_names


class TopicData(TypedDict):
    vocab: np.ndarray
    document_term_matrix: np.ndarray
    document_topic_matrix: np.ndarray
    topic_term_matrix: np.ndarray
    document_names: List[str]
    document_representation: np.ndarray
    corpus: List[str]
    transform: Optional[Callable]
    topic_names: List[str]
    group_labels: Optional[List[str]]


def prepare_pipeline_data(
    pipeline: Pipeline,
    corpus: Iterable[str],
    representation: Literal["term", "topic"] = "term",
) -> Dict:
    """Transforms corpus with the topic model, and extracts important matrices."""
    vectorizer, topic_model = split_pipeline(None, None, pipeline)
    try:
        print("Inferring topical content for documents.")
        document_topic_matrix = pipeline.transform(corpus)
    except (NotFittedError, AttributeError):
        print("Fitting pipeline, as it was not or cannot be prefitted.")
        document_topic_matrix = pipeline.fit_transform(corpus)
    document_term_matrix = vectorizer.transform(corpus)
    topic_term_matrix = topic_model.components_
    vocab = vectorizer.get_feature_names_out()
    res = {
        "corpus": corpus,
        "document_term_matrix": document_term_matrix,
        "document_topic_matrix": document_topic_matrix,
        "document_representation": document_term_matrix
        if representation == "term"
        else document_topic_matrix,
        "vocab": vocab,
        "topic_term_matrix": topic_term_matrix,
    }
    try:
        pipeline.transform(["Something."])
        res["transform"] = pipeline.transform
    except AttributeError:
        res["transform"] = None
    return res


def prepare_contextual_data(
    contextual_model: TransformerMixin, corpus: Iterable[str]
) -> Dict:
    """Transform corpus with a given contextual model."""
    try:
        print("Inferring topical content for documents.")
        document_topic_matrix = contextual_model.transform(corpus)
    except (NotFittedError, AttributeError):
        print("Fitting pipeline, as it was not or cannot be prefitted.")
        document_topic_matrix = contextual_model.fit_transform(corpus)
    document_term_matrix = contextual_model.vectorizer.transform(corpus)
    topic_term_matrix = contextual_model.components_
    document_representation = contextual_model.encoder_.encode(corpus)
    vocab = contextual_model.vectorizer.get_feature_names_out()
    res = {
        "corpus": corpus,
        "document_term_matrix": document_term_matrix,
        "document_topic_matrix": document_topic_matrix,
        "document_representation": document_representation,
        "vocab": vocab,
        "topic_term_matrix": topic_term_matrix,
    }
    try:
        contextual_model.transform(["Something."])
        res["transform"] = contextual_model.transform
    except AttributeError:
        res["transform"] = None
    return res


def filter_nan_docs(topic_data: Dict) -> None:
    """Filters out documents, the topical content of which contains nans.
    NOTE: The function works in place.
    """
    nan_documents = np.isnan(topic_data["document_topic_matrix"]).any(axis=1)
    n_nan_docs = np.sum(nan_documents)
    if n_nan_docs:
        warn(
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
        topic_data["document_names"] = list(
            np.array(topic_data["document_names"])[~nan_documents]
        )
        if topic_data["group_labels"]:
            topic_data["group_labels"] = list(
                np.array(topic_data["group_labels"])[~nan_documents]
            )


def prepare_topic_data(
    pipeline: Optional[Pipeline],
    contextual_model: Optional[TransformerMixin],
    corpus: Iterable[str],
    document_names: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
    group_labels: Optional[List[str]] = None,
    representation: Literal["term", "topic"] = "term",
) -> TopicData:
    """Prepares data from a topic model, a corpus and data about that corpus.
    Fits models if necessary, transforms data and extracts vocab along with document
    representations and word content.

    Parameters
    ----------
    corpus: iterable of str
        List of all works in the corpus you intend to visualize.
    pipeline: Pipeline, default None
        Sklearn compatible pipeline, that has two components:
        a vectorizer and a topic model.
    contextual_model: TransformerMixin, default None
        Contextual topic model to be visualized in topicwizard.
    document_names: list of str, default None
        List of document names in the corpus, if not provided documents will
        be labeled 'Document <index>'.
    topic_names: list of str, default None
        List of topic names in the corpus, if not provided topic
        names will be inferred.
    group_labels: list of str or None, default None
        List of preexisting labels for the documents.
        You can pass it along if you have genre labels for example.
        In this case an additional page will get created with information
        about how these groups relate to topics and words in the corpus.
    representation: "term" or "topic", default "term"
        Specifies which representation to use to display documents.
        Only in effect when the model is not contextual.
    """
    corpus = list(corpus)
    n_documents = len(corpus)
    if document_names is None:
        document_names = [f"Document {i}" for i in range(n_documents)]
    if (pipeline is not None) and (contextual_model is None):
        topic_data = prepare_pipeline_data(
            pipeline, corpus, representation=representation
        )
    elif (pipeline is None) and (contextual_model is not None):
        topic_data = prepare_contextual_data(contextual_model, corpus)
    else:
        raise ValueError(
            "You can only specify a pipeline XOR a contextual model, not both."
        )
    topic_data["group_labels"] = group_labels
    topic_data["document_names"] = document_names
    filter_nan_docs(topic_data)
    if topic_names is None:
        topic_names = infer_topic_names(
            topic_data["vocab"], topic_data["topic_term_matrix"]
        )
    topic_data["topic_names"] = topic_names
    return TopicData(**topic_data)
