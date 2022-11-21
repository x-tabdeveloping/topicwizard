"""Main module with exposed interface elements"""
from typing import Any, Iterable, Optional, Tuple, Union
import pandas as pd
from sklearn.pipeline import Pipeline

from topicwizard.apps.topic import plot_topics_
from topicwizard.apps.document import plot_documents_
from topicwizard.utils.app import is_notebook

port = 8050


def plot_topics(
    pipeline: Union[Pipeline, Tuple[Any, Any], None] = None,
    corpus: Optional[pd.DataFrame] = None,
    texts: Optional[Union[str, Iterable[str]]] = None,
    *,
    vectorizer: Optional[Any] = None,
    topic_model: Optional[Any] = None,
    topic_names: Optional[Iterable[str]] = None,
    **kwargs,
):
    """Interactively plots all topics and related word importances.

    Parameters
    ----------
    pipeline: Pipeline or tuple, optional
        Topic pipeline, can be specified as sklearn pipeline or
        a tuple of vectorizer and topic model.
    corpus: DataFrame, optional
        Dataframe containing the corpus, possibly with metadata.
    texts: str or iterable of str
        Column name in corpus or iterable of the texts in the corpus, depending
        on whether corpus is specified or not.
    vectorizer, optional
        Sklearn compatible vectorizer.
        Only needed if pipeline is not specified.
    topic_model, optional
        Sklearn compatible topic model.
        Only needed if pipeline is not specified.
    topic_names: iterable of str, optional
        Names of the topics in the topic model.
    """
    global port
    plot_topics_(
        pipeline=pipeline,
        corpus=corpus,
        texts=texts,
        vectorizer=vectorizer,
        topic_model=topic_model,
        topic_names=topic_names,
        port=port,
        mode="bar",
        **kwargs,
    )
    port += 1


def plot_wordclouds(
    pipeline: Union[Pipeline, Tuple[Any, Any], None] = None,
    corpus: Optional[pd.DataFrame] = None,
    texts: Optional[Union[str, Iterable[str]]] = None,
    *,
    vectorizer: Optional[Any] = None,
    topic_model: Optional[Any] = None,
    topic_names: Optional[Iterable[str]] = None,
    **kwargs,
):
    """Interactively plots all topics and wordplots over most relevant words.

    Parameters
    ----------
    pipeline: Pipeline or tuple, optional
        Topic pipeline, can be specified as sklearn pipeline or
        a tuple of vectorizer and topic model.
    corpus: DataFrame, optional
        Dataframe containing the corpus, possibly with metadata.
    texts: str or iterable of str
        Column name in corpus or iterable of the texts in the corpus, depending
        on whether corpus is specified or not.
    vectorizer, optional
        Sklearn compatible vectorizer.
        Only needed if pipeline is not specified.
    topic_model, optional
        Sklearn compatible topic model.
        Only needed if pipeline is not specified.
    topic_names: iterable of str, optional
        Names of the topics in the topic model.
    """
    global port
    plot_topics_(
        pipeline=pipeline,
        corpus=corpus,
        texts=texts,
        vectorizer=vectorizer,
        topic_model=topic_model,
        topic_names=topic_names,
        port=port,
        mode="wordcloud",
        **kwargs,
    )
    port += 1


def plot_documents(
    pipeline: Union[Pipeline, Tuple[Any, Any], None] = None,
    corpus: Optional[pd.DataFrame] = None,
    texts: Optional[Union[str, Iterable[str]]] = None,
    names: Optional[Union[str, Iterable[str]]] = None,
    *,
    vectorizer: Optional[Any] = None,
    topic_model: Optional[Any] = None,
    topic_names: Optional[Iterable[str]] = None,
    **kwargs,
):
    global port
    plot_documents_(
        pipeline=pipeline,
        corpus=corpus,
        texts=texts,
        names=names,
        vectorizer=vectorizer,
        topic_model=topic_model,
        topic_names=topic_names,
        port=port,
        **kwargs,
    )
    port += 1
