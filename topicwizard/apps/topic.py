"""Contains dash app for plotting topics."""
import warnings
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ctx, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from sklearn.pipeline import Pipeline

from topicwizard.components import mini_switcher, relevance_slider
from topicwizard.plots.topic import all_topics_plot, topic_plot, wordcloud
from topicwizard.utils.app import (add_callbacks, get_app, init_callbacks,
                                   is_notebook)
from topicwizard.utils.prepare import (calculate_top_words,
                                       prepare_pipeline_data,
                                       prepare_topic_data,
                                       prepare_transformed_data)

warnings.filterwarnings("ignore")

# -----------------------
# Layout
# -----------------------


def _create_layout(topic_names: Iterable[str], fit_data: Dict, mode: str):
    is_wordcloud = mode == "wordcloud"
    aspect_ratio = " aspect-square " if is_wordcloud else ""
    layout = html.Div(
        id="topic_view",
        className="""
            flex-row items-stretch flex flex-1 w-full h-full fixed
        """,
        children=[
            dcc.Store(id="topic_names", data=topic_names),
            dcc.Store(id="current_topic", data=0),
            dcc.Store(
                id="fit_store",
                data=fit_data,
            ),
            dcc.Store(id="mode", data=mode),
            dcc.Graph(
                id="all_topics_plot",
                className="flex-auto basis-3/5 transition-all m-5 ",
                responsive=True,
                config=dict(scrollZoom=True),
                animate=True,
            ),
            dcc.Graph(
                id="current_topic_plot",
                className="flex-auto basis-2/5 transition-all m-5 mb-9"
                + aspect_ratio,
                responsive=True,
                animate=True,
                animation_options=dict(frame=dict(redraw=True)),
                config=dict(scrollZoom=is_wordcloud),
            ),
            mini_switcher,
            relevance_slider,
        ],
    )
    return layout


# -------------------------------
# Callbacks
# -------------------------------
callbacks, cb = init_callbacks()


@cb(
    Output("current_topic", "data"),
    State("current_topic", "data"),
    Input("next_topic", "n_clicks"),
    Input("prev_topic", "n_clicks"),
    Input("all_topics_plot", "clickData"),
)
def update_current_topic(
    current_topic: int,
    next_clicks: int,
    prev_clicks: int,
    plot_click_data: Dict,
) -> int:
    """Updates current topic in the store when one is selected."""
    if "fit_store" == ctx.triggered_id:
        return 0
    if "all_topics_plot" == ctx.triggered_id:
        if plot_click_data is None:
            raise PreventUpdate()
        # In theory multiple points could be selected with
        # multiple customdata elements, so we unpack the first element.
        point, *_ = plot_click_data["points"]
        topic_id, *_ = point["customdata"]
        return topic_id
    if not next_clicks and not prev_clicks:
        raise PreventUpdate()
    if ctx.triggered_id == "next_topic":
        return current_topic + 1
    elif ctx.triggered_id == "prev_topic":
        return current_topic - 1
    else:
        raise PreventUpdate()


@cb(
    Output("current_topic_plot", "figure"),
    Input("current_topic", "data"),
    Input("fit_store", "data"),
    Input("lambda_slider", "value"),
    State("mode", "data"),
)
def update_current_topic_plot(
    current_topic: int, fit_store: Dict, alpha: float, mode: str
) -> go.Figure:
    """Updates the plots about the current topic in the topic view
    when the current topic is changed or when a new model is fitted.
    """
    if current_topic is None or fit_store is None:
        raise PreventUpdate()
    if mode == "bar":
        top_n = 30
    else:
        top_n = 70
    top_words = calculate_top_words(
        topic_id=current_topic, top_n=top_n, alpha=alpha, **fit_store
    )
    if mode == "bar":
        return topic_plot(top_words)
    else:
        return wordcloud(top_words)


@cb(
    Output("all_topics_plot", "figure"),
    Input("fit_store", "data"),
    Input("topic_names", "data"),
    Input("current_topic", "data"),
)
def update_all_topics_plot(
    fit_data: Dict,
    topic_names: List[str],
    current_topic: int,
) -> go.Figure:
    """Updates the topic overview plot when the fit, the topic names or the
    current topic change."""
    if not topic_names:
        # If there's missing data, prevent update.
        raise PreventUpdate()
    x, y = fit_data["topic_pos"]
    size = fit_data["topic_frequency"]
    topic_id = np.arange(fit_data["n_topics"])
    topic_data = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "size": size,
            "topic_id": topic_id,
        }
    )
    # Mapping topic names over to topic ids with a Series
    # since Series also function as a mapping, you can use them in the .map()
    # method
    names = pd.Series(topic_names)
    topic_data = topic_data.assign(topic_name=topic_data.topic_id.map(names))
    fig = all_topics_plot(topic_data, current_topic)
    return fig


@cb(
    Output("next_topic", "disabled"),
    Output("prev_topic", "disabled"),
    Input("current_topic", "data"),
    State("topic_names", "data"),
)
def update_topic_switcher(current_topic: int, topic_names: List[str]):
    """Updates the topic switcher component when the current topic changes."""
    if topic_names is None:
        raise PreventUpdate
    n_topics = len(topic_names)
    prev_disabled = current_topic == 0
    next_disabled = current_topic == n_topics - 1
    return next_disabled, prev_disabled


# ------------------------------------
# Main
# ------------------------------------


def plot_topics_(
    pipeline: Union[Pipeline, Tuple[Any, Any], None] = None,
    corpus: Optional[pd.DataFrame] = None,
    texts: Optional[Union[str, Iterable[str]]] = None,
    *,
    vectorizer: Optional[Any] = None,
    topic_model: Optional[Any] = None,
    topic_names: Optional[Iterable[str]] = None,
    mode: Literal["bar", "wordcloud"],
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
    mode: {'bar', 'wordcloud'}
        Indicates which type of plot should be used to display most relevant
        words.
    """
    if pipeline is None:
        if vectorizer is None or topic_model is None:
            raise TypeError(
                "You either have to specify a pipeline or a vectorizer "
                "and a topic model separately."
            )
    if texts is None and corpus is not None:
        raise TypeError(
            "You have to specify which column in the corpus"
            "should be used as texts."
        )
    if corpus is None and texts is None:
        raise TypeError("Either corpus or texts has to be specified.")
    if isinstance(pipeline, Pipeline):
        (_, vectorizer), (_, topic_model) = pipeline.steps
    if isinstance(pipeline, tuple):
        vectorizer, topic_model = pipeline
    if corpus is not None:
        texts = corpus[texts]
    assert topic_model is not None
    assert vectorizer is not None
    assert texts is not None
    n_topics = topic_model.n_components
    if topic_names is None:
        topic_names = [f"Topic {i_topic}" for i_topic in range(n_topics)]
    pipeline_data = prepare_pipeline_data(vectorizer, topic_model)
    transformed_data = prepare_transformed_data(vectorizer, topic_model, texts)
    topic_data = prepare_topic_data(**transformed_data, **pipeline_data)
    fit_data = {
        **pipeline_data,
        **topic_data,
    }
    app = get_app()
    app.layout = _create_layout(
        topic_names=topic_names, fit_data=fit_data, mode=mode
    )
    add_callbacks(app, callbacks)
    if is_notebook():
        kwargs["mode"] = "inline"
    app.run_server(**kwargs)
