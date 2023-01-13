"""Module containing plotting utilities for documents."""
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def documents_plot_3d(document_data: pd.DataFrame) -> go.Figure:
    """Plots all documents in 3D space, colors them according to dominant topic.

    Parameters
    ----------
    document_data: DataFrame
        Data about document position, topic and metadata.

    Returns
    -------
    Figure
        3D Scatter plot of all documents.
    """
    fig = px.scatter_3d(
        document_data,
        x="x",
        y="y",
        z="z",
        color="topic_name",
        custom_data=["doc_id"],
    )
    # fig.update_traces(hovertemplate=None, hoverinfo="none")
    fig.update_traces(
        hovertemplate="""
            %{customdata[0]} - %{customdata[1]}<br>
            <i>Click to select</i>
        """
    )
    axis = dict(
        showgrid=True,
        zeroline=True,
        visible=False,
    )
    fig.update_layout(
        # clickmode="event",
        # uirevision=True,
        modebar_remove=["lasso2d", "select2d"],
        hovermode="closest",
        paper_bgcolor="rgba(1,1,1,0)",
        plot_bgcolor="rgba(1,1,1,0)",
        hoverlabel=dict(font_size=11),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis),
    )
    return fig


def documents_plot(
    document_data: pd.DataFrame,
    selected: Optional[int] = None,
) -> go.Figure:
    """Plots all documents in 2D space, colors them according to dominant topic.

    Parameters
    ----------
    document_data: DataFrame
        Data about document position, topic and metadata.

    Returns
    -------
    Figure
        Scatter plot of all documents.
    """
    fig = px.scatter(
        document_data,
        render_mode="webgl",
        x="x",
        y="y",
        color="topic_name",
        custom_data=["doc_id"],
        hover_data={"x": False, "y": False, "name": True},
    )
    fig.update_xaxes(
        showgrid=True,
        zeroline=True,
        visible=False,
    )
    fig.update_yaxes(
        showgrid=True,
        zeroline=True,
        visible=False,
    )
    fig.update_layout(
        modebar_remove=["lasso2d", "select2d"],
        hovermode="closest",
        paper_bgcolor="rgba(1,1,1,0)",
        plot_bgcolor="rgba(1,1,1,0)",
        hoverlabel=dict(font_size=11),
        dragmode="pan",
    )
    if selected is not None:
        selected_point = document_data.set_index("doc_id").loc[int(selected)]
        fig.add_annotation(
            x=selected_point.x,
            y=selected_point.y,
            showarrow=True,
            text="",
            arrowwidth=2,
            arrowhead=2,
        )
    return fig


def document_topic_plot(
    topic_importances: pd.DataFrame,
    topic_names: List[str],
) -> go.Figure:
    """Plots topic importances for a selected document.

    Parameters
    ----------
    topic_importances: dict of int to float
        Mapping of topic id's to importances.
    topic_names: list of str
        List of topic names.

    Returns
    -------
    Figure
        Pie chart of topic importances for each document.
    """
    name_mapping = pd.Series(topic_names)
    topic_importances = topic_importances.assign(
        topic_name=topic_importances.topic_id.map(name_mapping)
    )
    fig = px.pie(
        topic_importances,
        values="importance",
        names="topic_name",
        color_discrete_sequence=px.colors.sequential.RdBu,
    )
    fig.update_traces(textposition="inside", textinfo="label")
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(1,1,1,0)",
        plot_bgcolor="rgba(1,1,1,0)",
    )
    return fig
