import os
import random
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from turftopic import KeyNMF, SemanticSignalSeparation

from topicwizard import figures
from topicwizard.pipeline import make_topic_pipeline


def test_figures():
    """Trains a variety of topic models and tests whether the figures can be produced for them."""
    newsgroups = fetch_20newsgroups(
        subset="all",
        categories=[
            "alt.atheism",
        ],
        remove=("headers", "footers", "quotes"),
    )
    texts = newsgroups.data
    labels = list(np.array(newsgroups.target_names)[newsgroups.target])
    trf = SentenceTransformer(
        "sentence-transformers/average_word_embeddings_glove.6B.300d"
    )
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
    example_document = "Joe Biden takes over presidential office from Donald Trump."
    plots = {
        "group_map": partial(figures.group_map, group_labels=labels),
        "group_topic_barcharts": partial(
            figures.group_topic_barcharts, group_labels=labels
        ),
        "group_wordclouds": partial(figures.group_wordclouds, group_labels=labels),
        "document_map": figures.document_map,
        "document_topic_timeline": partial(
            figures.document_topic_timeline,
            document=example_document,
        ),
        "document_topic_distribution": partial(
            figures.document_topic_distribution,
            documents=example_document,
        ),
        "topic_barcharts": figures.topic_barcharts,
        "topic_map": figures.topic_map,
        "topic_wordclouds": figures.topic_wordclouds,
        "word_map": figures.word_map,
        "word_association_barchart": partial(
            figures.word_association_barchart, words=["jesus", "allah"]
        ),
    }
    with tempfile.TemporaryDirectory() as dir_name:
        out_dir = Path(dir_name)
        for model_name, model in models.items():
            model_dir = out_dir.joinpath(model_name)
            model_dir.mkdir(exist_ok=True, parents=True)
            try:
                topic_data = model.prepare_topic_data(texts, embeddings=embeddings)
            except TypeError:
                topic_data = model.prepare_topic_data(texts)
            random_plots = random.sample(list(plots.keys()), 3)
            for plot_name in random_plots:
                plot_path = model_dir.joinpath(f"{plot_name}.html")
                fig = plots[plot_name](topic_data)
                fig.write_html(plot_path)
                assert os.path.isfile(plot_path)
