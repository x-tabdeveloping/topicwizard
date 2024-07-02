from typing import Dict

from topicwizard.data import TopicData
from topicwizard.prepare.documents import document_positions
from topicwizard.prepare.topics import topic_positions
from topicwizard.prepare.words import word_positions


def precompute_positions(data: Dict) -> TopicData:
    """Adds document, word and topic positions to TopicData objects,
    so that they do not have to be computed when the server starts.
    Great for cold starts and deployment to low-resource environments.

    Parameters
    ----------
    data: TopicData
        Original TopicData object without positions.

    Returns
    -------
    TopicData
        New topic data object with positions added.
    """
    new_data = {**data}
    if "topic_positions" not in new_data:
        new_data["topic_positions"] = topic_positions(data["topic_term_matrix"])
    if "document_positions" not in new_data:
        new_data["document_positions"] = document_positions(
            data["document_representation"]
        )
    if "word_positions" not in new_data:
        new_data["word_positions"] = word_positions(data["topic_term_matrix"])
    return TopicData(**new_data)
