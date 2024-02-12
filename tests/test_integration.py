import os
import random
import tempfile
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from turftopic import KeyNMF, SemanticSignalSeparation

from topicwizard import get_dash_app
from topicwizard.pipeline import make_topic_pipeline


def test_app_integration():
    """Trains a variety of topic models and tests whether the app can be produced based on them."""
    newsgroups = fetch_20newsgroups(
        subset="all",
        categories=[
            "misc.forsale",
            "sci.med",
            "comp.graphics",
            "alt.atheism",
            "talk.politics.misc",
        ],
        remove=("headers", "footers", "quotes"),
    )
    texts = newsgroups.data
    trf = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.asarray(trf.encode(texts))
    models = dict()
    models["nmf"] = make_topic_pipeline(
        CountVectorizer(stop_words="english", max_features=8000),
        NMF(10),
    ).fit(texts)
    models["lsa"] = make_topic_pipeline(
        TfidfVectorizer(stop_words="english", max_features=8000),
        TruncatedSVD(10),
    ).fit(texts)
    models["s3"] = SemanticSignalSeparation(10)
    models["keynmf"] = KeyNMF(10)
    for model_name, model in models.items():
        try:
            topic_data = model.prepare_topic_data(texts, embeddings=embeddings)
        except TypeError:
            topic_data = model.prepare_topic_data(texts)
        app = get_dash_app(topic_data, exclude_pages=set())
