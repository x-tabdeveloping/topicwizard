"""App for displaying documents in a corpus."""

from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dash import dcc, html, ctx
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
from topicwizard.components.document_inspector import document_inspector
from topicwizard.utils.app import (
    add_callbacks,
    get_app,
    init_callbacks,
    is_notebook,
)
from topicwizard.plots.documents import documents_plot, document_topic_plot
from topicwizard.utils.prepare import (
    calculate_top_words,
    prepare_pipeline_data,
    prepare_topic_data,
    prepare_transformed_data,
    prepare_document_data,
)
from topicwizard.components.accordion import accordion_callbacks

# -----------------------
# Layout
# -----------------------


def _create_layout(topic_names: Iterable[str], fit_data: Dict):
    layout = html.Div(
        id="documents_view",
        className="""
            flex-row items-stretch flex flex-1 w-full h-full fixed
            space-x-2
        """,
        children=[
            dcc.Store(id="topic_names", data=topic_names),
            dcc.Store(
                id="fit_store",
                data=fit_data,
            ),
            dcc.Graph(
                id="all_documents_plot",
                className="flex-1 basis-1/2 transition-all ",
                responsive=True,
                config=dict(scrollZoom=True),
            ),
            document_inspector,
        ],
    )
    return layout


# -------------------------------
# Callbacks
# -------------------------------
callbacks, cb = init_callbacks()
callbacks.extend(accordion_callbacks)


@cb(
    Output("document_selector", "options"),
    Input("fit_store", "data"),
)
def update_document_selector(fit_store: Dict) -> List[Dict]:
    if fit_store is None:
        raise PreventUpdate
    document_data = pd.DataFrame(fit_store["document_data"])
    return [
        {"label": doc_name, "value": doc_id}
        for doc_id, doc_name in zip(
            document_data["doc_id"], document_data["name"]
        )
    ]


@cb(
    Output("all_documents_plot", "figure"),
    Input("fit_store", "data"),
    Input("topic_names", "data"),
    Input("document_selector", "value"),
)
def update_all_documents_plot(
    fit_store: Dict, topic_names: List[str], selected: Optional[int]
):
    if fit_store is None or topic_names is None:
        raise PreventUpdate
    document_data = pd.DataFrame(fit_store["document_data"])
    mapping = pd.Series(topic_names)
    document_data["topic_name"] = document_data["topic_id"].map(mapping)
    fig = documents_plot(document_data, selected=selected)
    if ctx.triggered_id == "document_selector":
        fig.update_layout(uirevision=True)
    return fig


@cb(
    Output("document_selector", "value"),
    Input("all_documents_plot", "clickData"),
)
def select_document(selected_points: Dict) -> int:
    if not selected_points:
        raise PreventUpdate()
    point, *_ = selected_points["points"]
    print(point["customdata"])
    text_id = point["customdata"][0]
    return int(text_id)


@cb(
    Output("document_topics_graph", "figure"),
    Output("document_content", "children"),
    Input("document_selector", "value"),
    State("fit_store", "data"),
    State("topic_names", "data"),
)
def update_document_inspector(
    doc_id: int,
    fit_data: Dict,
    topic_names: List[str],
) -> Tuple[go.Figure, str]:
    if doc_id is None:
        raise PreventUpdate()
    doc_id = int(doc_id)
    document_data = (
        pd.DataFrame(fit_data["document_data"]).set_index("doc_id").loc[doc_id]
    )
    importances = pd.DataFrame(fit_data["document_topic_importance"])
    importances = importances[importances.doc_id == doc_id]
    fig = document_topic_plot(importances, topic_names)
    return (fig, document_data.text)


# -----------------------
# Main
# -----------------------


def plot_documents_(
    pipeline: Union[Pipeline, Tuple[Any, Any], None] = None,
    corpus: Optional[pd.DataFrame] = None,
    texts: Optional[Union[str, Iterable[str]]] = None,  # has to be repeatable
    names: Optional[Union[str, Iterable[str]]] = None,
    *,
    vectorizer: Optional[Any] = None,
    topic_model: Optional[Any] = None,
    topic_names: Optional[Iterable[str]] = None,
    **kwargs,
):
    # Checking if parameters are valid
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
    # Unpacking parameters
    if isinstance(pipeline, Pipeline):
        (_, vectorizer), (_, topic_model) = pipeline.steps
    if isinstance(pipeline, tuple):
        vectorizer, topic_model = pipeline
    if isinstance(texts, str):
        assert corpus is not None
        texts = corpus[texts]
    if isinstance(names, str):
        assert corpus is not None
        names = corpus[names]
    if names is None:
        assert texts is not None
        names = [f"Document {i_doc}" for i_doc, _ in enumerate(texts)]
    # If they didn't supply a corpus variable we create one
    if corpus is None:
        corpus = pd.DataFrame(
            dict(
                text=texts,
                name=names,
            )
        )
    else:
        corpus = corpus.assign(text=texts, name=names)
    # These are mainly here to satisfy the type checker
    assert topic_model is not None
    assert vectorizer is not None
    assert texts is not None
    n_topics = topic_model.n_components
    # Naming topics if they didn't supply topic names
    if topic_names is None:
        topic_names = [f"Topic {i_topic}" for i_topic in range(n_topics)]
    # Tranformations
    pipeline_data = prepare_pipeline_data(vectorizer, topic_model)
    transformed_data = prepare_transformed_data(vectorizer, topic_model, texts)
    topic_data = prepare_topic_data(**transformed_data, **pipeline_data)
    document_data = prepare_document_data(
        corpus=corpus, **transformed_data, **pipeline_data
    )
    fit_data = {
        **pipeline_data,
        **topic_data,
        **document_data,
    }
    # Creating application
    app = get_app()
    app.layout = _create_layout(topic_names=topic_names, fit_data=fit_data)
    add_callbacks(app, callbacks)
    if is_notebook():
        kwargs["mode"] = "inline"
    app.run_server(**kwargs)
