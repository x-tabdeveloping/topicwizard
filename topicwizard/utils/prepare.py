"""Utils for preparing topic models and corpuses to be plotted"""
from typing import Any, Dict, Iterable, List, Tuple

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import scipy.sparse as spr
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import pairwise_distances

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ModuleNotFoundError:
    print("Intel Sklearn extension could not be found continuing without accelaration")


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
    print("Transforming data with model")
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
    print("Preparing topic data")
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


def reduce_manifold_2d(embeddings: np.ndarray) -> np.ndarray:
    """Reduces embeddings to 2d with UMAP. SVD and standard scaler is added
    to the pipeline for speedup.

    Parameters
    ----------
    embeddings: ndarray of shape (n_observations, n_features)
        Embeddings to reduce.

    Returns
    -------
    ndarray of shape (n_observations, 2)
        Reduced embeddings.
    """
    dim_red_pipeline = Pipeline(
        [
            ("SVD", TruncatedSVD(20)),
            ("Scaler", StandardScaler()),
            (
                "UMAP",
                umap.UMAP(
                    n_components=2,
                    n_epochs=200,
                    n_neighbors=50,
                    min_dist=0.01,
                ),
            ),
        ]
    )
    return dim_red_pipeline.fit_transform(embeddings)


def reduce_pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Reduces embeddings to 2d with PCA. NMF is used to densify the matrix
    before PCA.

    Parameters
    ----------
    embeddings: ndarray of shape (n_observations, n_features)
        Embeddings to reduce.

    Returns
    -------
    ndarray of shape (n_observations, 2)
        Reduced embeddings.
    """
    dim_red_pipeline = Pipeline(
        [
            ("NMF", NMF(100)),
            ("Scaler", StandardScaler()),
            ("PCA", PCA(n_components=2)),
        ]
    )
    return dim_red_pipeline.fit_transform(embeddings)


def prepare_document_data(
    corpus: pd.DataFrame,
    document_term_matrix: np.ndarray,
    document_topic_matrix: np.ndarray,
    **kwargs,
) -> Dict:
    """Prepares document data for plotting"""
    print("Preparing documents")
    print(" - Obtaining dominant topics")
    dominant_topic = np.argmax(document_topic_matrix, axis=1)
    # Setting up dimensionality reduction pipeline
    # Calculating positions in 2D space
    print(" - Reducing document dimensionality")
    x, y = reduce_manifold_2d(document_term_matrix).T
    documents = corpus.assign(
        x=x,
        y=y,
        doc_id=np.arange(len(corpus.index)),
        topic_id=dominant_topic,
    )
    print(" - Calculating topic importances")
    importance_sparse = topic_document_importance(document_topic_matrix)
    return {
        "document_data": documents.to_dict(),
        "document_topic_importance": importance_sparse,
    }


def prepare_word2vec(model: Word2Vec) -> Dict:
    """Extracts embeddings and vocab from word2vec model"""
    embeddings = model.wv.vectors
    vocab = model.wv.index_to_key
    return dict(word_embeddings=embeddings, word_embedding_vocab=vocab)


def get_closest_words(
    embeddings: np.ndarray, n_closest: int = 5, distance_metric: str = "cosine"
) -> np.ndarray:
    """Finds n closest words to each word in the vocabulary.

    Parameters
    ----------
    embeddings: ndarray of shape (n_vocab, n_features)
        Matrix of all word embeddings.
    n_closest: int, default 5
        Number of closest word to find.
    distance_metric: str, default 'cosine'
        Distance metric to measure word distance.

    Returns
    -------
    ndarray of shape (n_vocab, n_closest)
        Indices of closest word for each word in the vocabulary.
    """
    # Calculates distance matrix with the given metric
    distance_matrix = pairwise_distances(embeddings, metric=distance_metric)
    # Partitions array so that the smallest k elements along axis 1 are at the
    # lowest k dimensions, then I slice the array to only get the top indices
    # We do plus 1, as obviously the closest word is gonna be the word itself
    closest = np.argpartition(distance_matrix, kth=n_closest + 1, axis=1)[
        :, 1 : n_closest + 1
    ]
    return closest


def prepare_word_data(
    word_embeddings: np.ndarray,
    word_embedding_vocab: Iterable[str],
    document_term_matrix: np.ndarray,
    components: np.ndarray,
    vocab: np.ndarray,
    **kwargs,
) -> Dict:
    """Prepares word data for plotting"""
    # Reducing dimensionality, so embeddings can be visualized in 2d
    x, y = reduce_manifold_2d(word_embeddings).T
    # Calculating word frequency over the entire corpus
    word_freqs = document_term_matrix.sum(axis=0).A1
    # Mapping terms to their frequencies
    # This is important because word embedding models might have different
    # vocabulary from the vectorizer of the topic model.
    freq_dict = {term: frequency for term, frequency in zip(vocab, word_freqs)}
    # This maps the frequencies over to the word embedding vocab
    word_importance = pd.Series(word_embedding_vocab).map(freq_dict)
    # Calculating most important topic for each word
    dominant_topic = np.argmax(components, axis=0)
    # Mapping terms to dominant topics
    topic_dict = {term: topic for term, topic in zip(vocab, dominant_topic)}
    # Mapping dominant topics to the word embedding model's vocab
    topic = pd.Series(word_embedding_vocab).map(topic_dict)
    # Number of words in the word embedding model
    n_embedding_vocab, _ = word_embeddings.shape
    # Calculating 5 closest words for each word
    closest_words = get_closest_words(word_embeddings, n_closest=5).tolist()
    # Creating a dataframe out of all data for the words
    words = pd.DataFrame(
        dict(
            word_id=np.arange(n_embedding_vocab),
            word=word_embedding_vocab,
            x=x,
            y=y,
            frequency=word_importance,
            topic_id=topic,
            closest_words=closest_words,
        )
    )
    words = words.dropna()
    return {"word_data": words.to_dict()}


def semantic_kernel(
    words: pd.DataFrame, word_id: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Computes semantic kernel for the given word with two levels of
    assocation based on the closest words.

    Parameters
    ----------
    words: DataFrame
        Dataframe containing precomputed word data.
    word_id: int
        Id of the word to calculate the kernel for.

    Returns
    -------
    nodes: DataFrame
        Dataframe containing all the associated words.
    edges: ndarray of shape (n_edges, 2)
        Edges between words in the semantic graph.
    """
    # I'm using a dict to represent the kernel, because of the O(1) lookup time
    # this will make it easy for me to maintain unique items.
    # The kernel will be represented as a mapping of word indices to their
    # association levels.
    kernel = {word_id: 0}
    # Edges in the graph will be represented with tuples of word indices
    edges = []
    # I set word id to be the index of the table as I intend to index and join
    # it based on word ids.
    words = words.set_index("word_id")
    seed = words.loc[word_id]
    first_level_assoc = seed.closest_words
    for first_level_word in first_level_assoc:
        # Adding all first level association words to the kernel
        kernel[first_level_word] = 1
        edges.append((word_id, first_level_word))
        second_level_assoc = words.loc[first_level_word]
        for second_level_word in second_level_assoc:
            # Adding all second level association words to the kernel
            edges.append((second_level_word, first_level_word))
            if second_level_word not in kernel:
                kernel[second_level_word] = 2
    # Converting the dict to a Series, so the index becomes the word ids
    # then converting it into a DataFrame, so we can join it with the rest
    # of the data for the words.
    nodes = pd.Series(kernel).to_frame(name="association_level")
    # Joining, so I won't have to when I wanna plot things.
    nodes = nodes.join(words[["word", "x", "y"]])
    return nodes, np.array(edges)
