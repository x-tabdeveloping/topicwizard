from typing import Any, Iterable, Optional, Tuple, Union
import warnings

import dash
import pandas as pd
from sklearn.pipeline import Pipeline

from topicwizard.apps.topic import plot_topics_


# Checking if code is running in a jupyter notebook or not
def is_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__  # noqa
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    try:
        from jupyter_dash import JupyterDash

        Dash = JupyterDash
    except ImportError:
        warnings.warn(
            "You are running code in a Jupyter notebook, but you don't have "
            "JupyterDash installed, if you wish to use topic wizard inside "
            "Jupyter install jupyter_dash. Defaulting to server mode."
        )
        Dash = dash.Dash
else:
    Dash = dash.Dash

port = 8050


def plot_topics(
    pipeline: Union[Pipeline, Tuple[Any, Any], None] = None,
    corpus: Optional[pd.DataFrame] = None,
    texts: Optional[Union[str, Iterable[str]]] = None,
    *,
    vectorizer: Optional[Any] = None,
    topic_model: Optional[Any] = None,
    topic_names: Optional[Iterable[str]] = None,
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
    global Dash
    extra_kwargs = {}
    if is_notebook():
        extra_kwargs["mode"] = "inline"
    plot_topics_(
        pipeline=pipeline,
        corpus=corpus,
        texts=texts,
        vectorizer=vectorizer,
        topic_model=topic_model,
        topic_names=topic_names,
        port=port,
        dash=Dash,
        extra_kwargs=extra_kwargs
    )
    port += 1
