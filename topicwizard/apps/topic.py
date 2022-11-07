"""Contains dash app for plotting topics."""
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ctx, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from sklearn.pipeline import Pipeline

from topicwizard.plots.topic import all_topics_plot, topic_plot
from topicwizard.utils.prepare import (calculate_top_words,
                                       prepare_pipeline_data,
                                       prepare_topic_data,
                                       prepare_transformed_data)
warnings.filterwarnings("ignore")

# -------------------------------
# Callbacks
# -------------------------------


callbacks = []


def cb(*args, **kwargs) -> Callable:
    """Decorator to add a function to the global callback list"""

    def _cb(func: Callable):
        callbacks.append({"function": func, "args": args, "kwargs": kwargs})
        return func

    return _cb


def add_callbacks(app: dash.Dash) -> None:
    """Adds the list of callbacks to a Dash app."""
    for callback in callbacks:
        app.callback(*callback["args"], **callback["kwargs"])(
            callback["function"]
        )


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
)
def update_current_topic_plot(
    current_topic: int, fit_store: Dict, alpha: float
) -> go.Figure:
    """Updates the plots about the current topic in the topic view
    when the current topic is changed or when a new model is fitted.
    """
    if current_topic is None or fit_store is None:
        raise PreventUpdate()
    top_words = calculate_top_words(
        topic_id=current_topic, top_n=30, alpha=alpha, **fit_store
    )
    return topic_plot(top_words)


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


# -----------------------
# Layout
# -----------------------

button_class = """
    text-xl transition-all ease-in 
    justify-center content-center items-center
    text-gray-500 hover:text-sky-600
    flex flex-1
"""

mini_switcher = html.Div(
    className="""
        fixed flex flex-none flex-row justify-center content-middle
        left-0.5 bottom-10 h-16 w-32 bg-white shadow rounded-full
        rounded-full ml-5
    """,
    children=[
        html.Button(
            "<-",
            id="prev_topic",
            title="Switch to previous topic",
            className=button_class,
        ),
        html.Button(
            "->",
            id="next_topic",
            title="Switch to next topic",
            className=button_class,
        ),
    ],
)

relevance_slider = html.Div(
    className="""
        fixed flex flex-none flex-row justify-between items-center
        left-40 bottom-10 h-16 w-96 bg-white shadow rounded-full/
        rounded-full ml-5 px-6 py-6
    """,
    children=[
        html.Div("λ :", className="text-xl text-gray-500"),
        dcc.Slider(
            id="lambda_slider",
            value=1.0,
            min=0.0,
            max=1.0,
            className="flex-1 mt-5",
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ],
)


def _create_layout(topic_names: Iterable[str], fit_data: Dict):
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
            dcc.Graph(id="all_topics_plot", className="flex-1 basis-3/5 "),
            dcc.Graph(id="current_topic_plot", className="flex-1 basis-2/5 "),
            mini_switcher,
            relevance_slider,
        ],
    )
    return layout


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
    port: int = 8050,
    dash=dash.Dash,
    use_reloader: bool = False,
    extra_kwargs: Dict = {},
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
    app = dash(
        __name__,
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
    app.layout = _create_layout(topic_names=topic_names, fit_data=fit_data)
    add_callbacks(app)
    app.run_server(
        debug=True, use_reloader=use_reloader, port=port, **extra_kwargs
    )
