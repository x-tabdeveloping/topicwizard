from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from dash_extensions.enrich import DashBlueprint, html

from topicwizard.data import TopicData
from topicwizard.model_interface import TopicModel

BlueprintCreator = Callable[..., DashBlueprint]


def create_blank_page(name: str) -> DashBlueprint:
    blueprint = DashBlueprint()
    blueprint.layout = html.Div(id=f"{name}_container")
    return blueprint


def prepare_blueprint(
    create_blueprint: BlueprintCreator,
    topic_data: TopicData,
    document_names: Optional[List[str]] = None,
    group_labels: Optional[List[str]] = None,
    *args,
    **kwargs,
) -> DashBlueprint:
    blueprint = create_blueprint(
        *args,
        **topic_data,
        document_names=document_names,
        group_labels=group_labels,
        **kwargs,
    )
    return blueprint
