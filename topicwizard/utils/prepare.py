"""Utils for preparing topic models and corpuses to be plotted"""
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import scipy.sparse as spr
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE


def word_relevance(components: np.ndarray, alpha: float) -> np.ndarray:
    """Returns relevance scores for each topic for each word.

    Parameters
    ----------
    components: ndarray of shape (n_topics, n_vocab)
        Topic word probability matrix.
    alpha: float
        Weight parameter.

    Returns
    -------
    ndarray of shape (n_topics, n_vocab)
        Topic word relevance matrix.
    """
    prior = np.sum(components, axis=0)
    relevance = alpha * np.log(components) + (1 - alpha) * np.log(
        components / np.sum(components, axis=0)
    )
    return relevance


def calculate_top_words(
    vectorizer: Any, topic_model: Any, top_n: int, alpha: float
) -> pd.DataFrame:
    """Arranges top N words of each topic to a DataFrame.

    Parameters
    ----------
    vectorizer
        Sklearn compatible vectorizer.
    topic_model
        Sklearn topic model.
    top_n: int
        Number of words to include for each topic.
    alpha: float
        Weight parameter.

    Returns
    -------
    DataFrame
        Top N words for each topic with importance scores.

    Note
    ----
    Importance scores are exclusively calculated from the topic model's
    'components_' attribute and do not have anything to do with the empirical
    distribution of words in each topic. This has to be kept in mind, as some models
    keep counts in their 'components_' attribute (e.g. DMM).
    Would probably be smart to reconsider this in the future.
    """
    relevance = topic_model.components_#word_relevance(topic_model.components_, alpha=alpha)
    overall_relevance = relevance.sum(axis=0)
    vocab = vectorizer.get_feature_names_out()
    n_topics = topic_model.n_components
    # Wrangling the data into tuple records
    res = pd.DataFrame(
        columns=["topic", "word", "importance", "overall_importance"]
    )
    for i_topic in range(n_topics):
        # Selecting N highest ranking feature indices for this topic
        highest = np.argpartition(relevance[i_topic], -top_n)[-top_n:]
        top_words = vocab[highest]
        top_relevance = relevance[i_topic, highest]
        top_overall = overall_relevance[highest]
        topic_res = pd.DataFrame(
            {
                "topic": i_topic,
                "word": top_words,
                "importance": top_relevance,
                "overall_importance": top_overall,
            }
        )
        res = pd.concat((res, topic_res), ignore_index=True)
    return res


# def calculate_topic_document_importance(
#     vectorizer: Any, topic_model: Any, texts: Iterable[str]
# ) -> Dict[int, Dict[int, float]]:
#     """Calculates topic importances for each document.
#
#     Parameters
#     ----------
#     vectorizer
#         Sklearn compatible vectorizer.
#     topic_model
#         Sklearn topic model.
#     corpus: DataFrame
#         Data frame containing the cleaned corpus with ids.
#
#     Returns
#     -------
#     dict of int to (dict of int to float)
#        Mapping of document ids to a dictionary of topic ids to importances.
#     """
#     pred = topic_model.transform(vectorizer.transform(texts))
#     lil = spr.lil_matrix(pred)
#     importance_list = []
#     for topic_ids, importances in zip(lil.rows, lil.data):
#         importance_list.append(
#             {
#                 topic_id: importance
#                 for topic_id, importance in zip(topic_ids, importances)
#             }
#         )
#     importance_dict = {
#         document_id: topics
#         for document_id, topics in zip(corpus.id_nummer, importance_list)
#     }
#     return importance_dict


def calculate_topic_data(
    vectorizer: Any, topic_model: Any, texts: Iterable[str]
) -> pd.DataFrame:
    """Calculates topic positions in 2D space as well as topic sizes
    based on their empirical importance in the corpus.

    Parameters
    ----------
    vectorizer
        Sklearn compatible vectorizer.
    topic_model
        Sklearn topic model.
    texts: iterable of str
        Texts in the corpus.

    Returns
    -------
    DataFrame
        Data about topic sizes and positions.
    """
    # Calculating topic predictions for each document.
    pred = topic_model.transform(vectorizer.transform(texts))
    _, n_topics = pred.shape
    # Calculating topic size from the empirical importance of topics
    size = pred.sum(axis=0)
    components = topic_model.components_
    # Calculating topic positions with t-SNE
    x, y = (
        TSNE(perplexity=5, init="pca", learning_rate="auto")
        .fit_transform(components)
        .T
    )
    return pd.DataFrame(
        {"topic_id": range(n_topics), "x": x, "y": y, "size": size}
    )
