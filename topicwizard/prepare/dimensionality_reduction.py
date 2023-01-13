"""Utilities for dimenstionality reduction."""
from typing import Literal

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import (
    NMF,
    TruncatedSVD,
    PCA,
)
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ModuleNotFoundError:
    print(
        "Intel Sklearn extension could not be found continuing without accelaration"
    )


def reduce_manifold_2d(
    embeddings: np.ndarray, which: Literal["umap", "tsne"]
) -> np.ndarray:
    """Reduces embeddings to 2d with UMAP or TSNE. SVD and standard scaler is added
    to the pipeline for speedup.

    Parameters
    ----------
    embeddings: ndarray of shape (n_observations, n_features)
        Embeddings to reduce.
    which: "umap" or "tsne"
        Indicates which manifold method to use.

    Returns
    -------
    ndarray of shape (n_observations, 2)
        Reduced embeddings.
    """
    if which == "umap":
        manifold = (
            umap.UMAP(
                n_components=2,
                n_epochs=200,
                n_neighbors=50,
                min_dist=0.01,
            ),
        )
    else:
        manifold = TSNE(n_components=2)
    dim_red_pipeline = Pipeline(
        [
            ("SVD", TruncatedSVD(20)),
            ("Scaler", StandardScaler()),
            ("Manifold", manifold),
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
