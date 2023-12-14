from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from dash_extensions.enrich import DashBlueprint, html
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

from topicwizard.prepare.data import prepare_topic_data

BlueprintCreator = Callable[..., DashBlueprint]


def create_blank_page(name: str) -> DashBlueprint:
    blueprint = DashBlueprint()
    blueprint.layout = html.Div(id=f"{name}_container")
    return blueprint


def prepare_blueprint(
    corpus: Iterable[str],
    model: Union[TransformerMixin, Pipeline],
    create_blueprint: BlueprintCreator,
    document_names: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
    group_labels: Optional[List[str]] = None,
    *args,
    **kwargs,
) -> DashBlueprint:
    topic_data = prepare_topic_data(
        model=model,
        corpus=corpus,
        document_names=document_names,
        topic_names=topic_names,
        group_labels=group_labels,
    )
    blueprint = create_blueprint(
        *args,
        model=model,
        **topic_data,
        **kwargs,
    )
    return blueprint
