import random
import warnings
from typing import Callable, Dict, List, Optional, Set

import dash_mantine_components as dmc
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    exceptions, html)

from topicwizard.data import TopicData

from .widget import Widget


def create_widget_container(
    widgets: List[Widget], topic_data: TopicData, app_id: Optional[str] = None
) -> DashBlueprint:
    # --------[ Collecting blueprints ]--------
    ids = []
    blueprints = []
    options = []
    for widget in widgets:
        try:
            widget.validate_data(topic_data)
        except TypeError as e:
            warnings.warn(f"Widget {widget.name} could not be created. Reason: {e}")
        blueprints.append(widget.create_blueprint(topic_data, app_id=app_id))
        component_id = f"{widget.id_prefix}_{app_id}"
        options.append({"value": component_id, "label": widget.name})
        ids.append(component_id)

    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()

    app_blueprint.layout = html.Div(
        [
            html.Div(
                [
                    dmc.SegmentedControl(
                        id=f"page_picker_{app_id}",
                        data=options,
                        value=ids[0],
                        color="blue",
                        size="md",
                        radius="sm",
                        mb=10,
                        fullWidth=True,
                    ),
                ],
                className="top-0 left-0 flex-none",
            ),
            html.Div(
                [blueprint.layout for blueprint in blueprints],
                className="flex flex-col flex-1",
            ),
        ],
        className="""
            w-full h-full flex-col flex items-stretch overflow-y-scroll
            bg-white
        """,
    )

    # --------[ Registering callbacks ]--------
    for blueprint in blueprints:
        blueprint.register_callbacks(app_blueprint)

    @app_blueprint.callback(
        *[Output(component_id, "className") for component_id in ids],
        Input(f"page_picker_{app_id}", "value"),
        prevent_initial_callback=False,
    )
    def switch_page(component_id: str):
        hidden = "hidden"
        visible = "flex flex-1 flex-col p-3"
        res = [hidden] * len(ids)
        for i_component in range(len(ids)):
            if component_id == ids[i_component]:
                res[i_component] = visible
                break
        print(res)
        return res

    return app_blueprint
