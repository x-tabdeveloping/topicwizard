"""Module containing plotting utilities for documents."""
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def documents_plot(document_data: pd.DataFrame) -> go.Figure:
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
        custom_data=[
            "værk",
            "forfatter",
            "group",
            "tlg_genre",
            "topic_name",
            "id_nummer",
        ],
    )
    # fig.update_traces(hovertemplate=None, hoverinfo="none")
    fig.update_traces(
        hovertemplate="""
            %{customdata[0]} - %{customdata[1]}<br>
            <i>Click to select</i>
        """
    )
    annotations = []
    for _index, row in document_data.iterrows():
        name = f"{row.værk} - {row.forfatter}"
        annotations.append(
            dict(
                x=row.x,
                y=row.y,
                z=row.z,
                text=name,
                bgcolor="white",
                bordercolor="black",
                arrowsize=1,
                arrowwidth=2,
                borderwidth=3,
                borderpad=10,
                font=dict(size=16, color="#0369a1"),
                visible=False,
                # clicktoshow="onout",
            )
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
        scene=dict(
            xaxis=axis, yaxis=axis, zaxis=axis, annotations=annotations
        ),
    )
    return fig


def document_topic_plot(
    topic_importances: Dict[int, float],
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
    importances = pd.DataFrame.from_records(
        list(topic_importances.items()), columns=["topic_id", "importance"]
    )
    importances = importances.assign(topic_id=importances.topic_id.astype(int))
    name_mapping = pd.Series(topic_names)
    importances = importances.assign(
        topic_name=importances.topic_id.map(name_mapping)
    )
    fig = px.pie(
        importances,
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
