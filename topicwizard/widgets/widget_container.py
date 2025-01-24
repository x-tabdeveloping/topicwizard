import random
import warnings
from typing import Callable, Dict, List, Optional, Set

import dash_mantine_components as dmc
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    exceptions, html)
from dash_iconify import DashIconify

from topicwizard.data import TopicData

from .widget import Widget


def create_widget_container(
    widgets: List[Widget], topic_data: TopicData, app_id: Optional[str] = None
) -> DashBlueprint:
    # --------[ Collecting blueprints ]--------
    ids = []
    names = []
    icons = []
    blueprints = []
    for widget in widgets:
        try:
            widget.validate_data(topic_data)
        except TypeError as e:
            warnings.warn(f"Widget {widget.name} could not be created. Reason: {e}")
        blueprints.append(widget.create_blueprint(topic_data, app_id=app_id))
        component_id = f"{widget.id_prefix}_{app_id}"
        ids.append(component_id)
        icons.append(widget.icon)
        names.append(widget.name)

    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()

    app_blueprint.layout = html.Div(
        dmc.Tabs(
            [
                dmc.TabsList(
                    [
                        dmc.Tab(name, icon=DashIconify(icon=icon), value=component_id)
                        for name, icon, component_id in zip(names, icons, ids)
                    ],
                    position="center",
                ),
                *[
                    dmc.TabsPanel(blueprint.layout, value=component_id)
                    for blueprint, component_id in zip(blueprints, ids)
                ],
            ],
            color="blue",
            orientation="horizontal",
            className="w-full h-full flex-1 overflow-y-scroll items-stretch flex flex-col p-8",
            value=ids[0],
            styles={
                "panel": {
                    "flex-grow": "1",
                    "flex-direction": "column",
                    "height": "100%",
                }
            },
        ),
        className="""
            w-full h-full flex-col flex items-stretch
            bg-white
        """,
    )

    return app_blueprint
