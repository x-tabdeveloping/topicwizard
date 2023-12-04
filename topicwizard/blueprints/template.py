from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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
    pipeline: Optional[Pipeline],
    contextual_model: Optional[TransformerMixin],
    corpus: Iterable[str],
    create_blueprint: BlueprintCreator,
    document_names: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
    group_labels: Optional[List[str]] = None,
    *args,
    **kwargs,
) -> DashBlueprint:
    topic_data = prepare_topic_data(
        pipeline=pipeline,
        contextual_model=contextual_model,
        corpus=corpus,
        document_names=document_names,
        topic_names=topic_names,
        group_labels=group_labels,
    )
    blueprint = create_blueprint(
        *args,
        contextual_model=contextual_model,
        pipeline=pipeline,
        **topic_data,
        **kwargs,
    )
    return blueprint
