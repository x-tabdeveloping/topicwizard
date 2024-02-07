from abc import abstractmethod
from typing import Any, List, Protocol

from topicwizard.data import TopicData


class TopicModel(Protocol):
    @abstractmethod
    def prepare_topic_data(
        self,
        corpus: List[str],
        *args: Any,
        **kwargs: Any,
    ) -> TopicData:
        ...
