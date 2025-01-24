from abc import ABC, abstractmethod
from typing import Sequence

import dash_mantine_components as dmc
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    html)
from dash_iconify import DashIconify

from topicwizard.data import TopicData, TopicDataAttribute


class Widget(ABC):
    """Base class for topicwizard widgets"""

    needed_attributes: Sequence[TopicDataAttribute]
    icon: str
    name: str
    id_prefix: str

    def create_label(self):
        return dmc.Center(
            [
                DashIconify(
                    icon=self.icon,
                    width=16,
                ),
                html.Span(self.name),
            ],
            style={"gap": 10},
        )

    @abstractmethod
    def create_blueprint(
        self, topic_data: TopicData, app_id: str = ""
    ) -> DashBlueprint:
        pass

    def validate_data(self, topic_data: TopicData):
        missing = []
        for attribute in self.needed_attributes:
            if topic_data.get(attribute, None) is None:
                missing.append(attribute)
        if not missing:
            return
        raise TypeError(
            f"Topic Data is missing the following attributes: {missing}, needed by Widget {self.name}"
        )
