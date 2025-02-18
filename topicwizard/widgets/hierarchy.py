import dash_mantine_components as dmc
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    html)
from turftopic.hierarchical import TopicNode

from topicwizard.data import TopicData

from .widget import Widget


def create_hierarchy_widget(
    app_id: str,
    hierarchy: TopicNode,
    **kwargs,
) -> DashBlueprint:
    # --------[ Preparing data ]--------
    fig = hierarchy.plot_tree()
    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            dmc.Center(
                dmc.Text(
                    """ This widget allows you to explore hierarchical structures in topic models.
                    Zoom to go closer to the individual nodes and hover over a node to see word importance scores in a node.
                    """,
                    size="sm",
                    fw=400,
                    c="dimmed",
                    className="pb-6",
                ),
            ),
            dcc.Graph(
                figure=fig, className="flex-1 flex", config=dict(scrollZoom=True)
            ),
        ],
        className="""
        flex flex-1 flex-col
        p-3 h-full
        """,
        id=f"hierarchy_{app_id}",
    )
    return app_blueprint


class TopicHierarchy(Widget):
    needed_attributes = ["hierarchy"]
    icon = "material-symbols:account-tree-outline"
    name = "Topic Hierarchy"
    id_prefix = "topic_hierarchy"

    def __init__(self):
        super().__init__()

    def create_blueprint(self, topic_data: TopicData, app_id: str = ""):
        return create_hierarchy_widget(app_id, **topic_data)
