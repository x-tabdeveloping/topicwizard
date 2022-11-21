"""Utils for preparing topic models and corpuses to be plotted"""
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import scipy.sparse as spr
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import normalize, StandardScaler


def min_max_norm(a) -> np.ndarray:
    """Performs min max normalization on an ArrayLike"""
    a = (a - np.min(a)) / (np.max(a) - np.min(a))
    return a


def word_relevance(
    topic_id: int,
    term_frequency: np.ndarray,
    topic_term_frequency: np.ndarray,
    alpha: float,
) -> np.ndarray:
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
    probability = np.log(topic_term_frequency[topic_id])
    probability[probability == -np.inf] = np.nan
    lift = np.log(topic_term_frequency[topic_id] / term_frequency)
    lift[lift == -np.inf] = np.nan
    relevance = alpha * probability + (1 - alpha) * lift
    return relevance


def calculate_top_words(
    topic_id: int,
    top_n: int,
    alpha: float,
    term_frequency: np.ndarray,
    topic_term_frequency: np.ndarray,
    vocab: np.ndarray,
    **kwargs,
) -> pd.DataFrame:
    """Arranges top N words by relevance for the given topic into a DataFrame."""
    vocab = np.array(vocab)
    term_frequency = np.array(term_frequency)
    topic_term_frequency = np.array(topic_term_frequency)
    relevance = word_relevance(
        topic_id, term_frequency, topic_term_frequency, alpha=alpha
    )
    highest = np.argpartition(-relevance, top_n)[:top_n]
    res = pd.DataFrame(
        {
            "word": vocab[highest],
            "importance": topic_term_frequency[topic_id, highest],
            "overall_importance": term_frequency[highest],
            "relevance": relevance[highest],
        }
    )
    return res


def prepare_pipeline_data(vectorizer: Any, topic_model: Any) -> Dict:
    """Prepares data about the pipeline for storing
    in local store and plotting"""
    n_topics = topic_model.n_components
    vocab = vectorizer.get_feature_names_out()
    components = topic_model.components_
    # Making sure components are normalized
    # (remember this is not necessarily the case with some models)
    components = normalize(components, norm="l1", axis=1)
    return {
        "n_topics": n_topics,
        "vocab": vocab.tolist(),
        "components": components.tolist(),
    }


def prepare_transformed_data(
    vectorizer: Any, topic_model: Any, texts: Iterable[str]
) -> Dict:
    """Runs pipeline on the given texts and returns the document term matrix
    and the topic document distribution."""
    # Computing doc-term matrix for corpus
    document_term_matrix = vectorizer.transform(texts)
    # Transforming corpus with topic model for empirical topic data
    document_topic_matrix = topic_model.transform(document_term_matrix)
    return {
        "document_term_matrix": document_term_matrix,
        "document_topic_matrix": document_topic_matrix,
    }


def prepare_topic_data(
    document_term_matrix: np.ndarray,
    document_topic_matrix: np.ndarray,
    components: np.ndarray,
    **kwargs,
) -> Dict:
    """Prepares data about topics for plotting."""
    components = np.array(components)
    # Calculating document lengths
    document_lengths = document_term_matrix.sum(axis=1)
    # Calculating an estimate of empirical topic frequencies
    topic_frequency = (document_topic_matrix.T * document_lengths).sum(axis=1)
    topic_frequency = np.squeeze(np.asarray(topic_frequency))
    # Calculating empirical estimate of term-topic frequencies
    # shape: (n_topics, n_vocab)
    topic_term_frequency = (components.T * topic_frequency).T
    # Empirical term frequency
    term_frequency = topic_term_frequency.sum(axis=0)
    term_frequency = np.squeeze(np.asarray(term_frequency))
    # Determining topic positions with TSNE
    topic_pos = (
        TSNE(perplexity=5, init="pca", learning_rate="auto").fit_transform(components).T
    )
    return {
        "topic_frequency": topic_frequency.tolist(),
        "topic_pos": topic_pos.tolist(),
        "term_frequency": term_frequency.tolist(),
        "topic_term_frequency": topic_term_frequency.tolist(),
    }


def topic_document_importance(
    document_topic_matrix: np.ndarray,
) -> Dict:
    """Calculates topic importances for each document."""
    coo = spr.coo_array(document_topic_matrix)
    topic_doc_imp = pd.DataFrame(
        dict(doc_id=coo.row, topic_id=coo.col, importance=coo.data)
    )
    return topic_doc_imp.to_dict()


def prepare_document_data(
    corpus: pd.DataFrame,
    document_term_matrix: np.ndarray,
    document_topic_matrix: np.ndarray,
    **kwargs,
) -> Dict:
    """Prepares document data for plotting"""
    dominant_topic = np.argmax(document_topic_matrix, axis=1)
    # Setting up dimensionality reduction pipeline
    dim_red_pipeline = Pipeline(
        [
            ("SVD", TruncatedSVD(20)),
            ("Scaler", StandardScaler()),
            # (
            #     "t-SNE",
            #     TSNE(2, perplexity=10, n_iter=300, init="pca", learning_rate="auto"),
            # ),
            (
                "UMAP",
                umap.UMAP(
                    n_components=2,
                    n_epochs=200,
                    n_neighbors=50,
                    # metric="cosine",
                    min_dist=0.01,
                ),
            ),
            # ("PCA", PCA(n_components=2)),
        ]
    )
    # Calculating positions in 2D space
    x, y = dim_red_pipeline.fit_transform(document_term_matrix).T
    documents = corpus.assign(
        x=x,
        y=y,
        doc_id=np.arange(len(corpus.index)),
        topic_id=dominant_topic,
    )
    importance_sparse = topic_document_importance(document_topic_matrix)
    return {
        "document_data": documents.to_dict(),
        "document_topic_importance": importance_sparse,
    }
