from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

import topicwizard
from topicwizard.compatibility import BERTopicWrapper, gensim_pipeline

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
corpus = newsgroups.data


def test_bertopic():
    model = BERTopicWrapper(BERTopic(language="english"))
    topic_data = model.prepare_topic_data(corpus)
    app = topicwizard.get_dash_app(topic_data)


# def test_gensim():
#     tokenized_corpus = [list(tokenize(text, lower=True)) for text in corpus]
#     dictionary = Dictionary(tokenized_corpus)
#     bow_corpus = [dictionary.doc2bow(text) for text in tokenized_corpus]
#     lda = LdaModel(bow_corpus, num_topics=10)
#     pipeline = gensim_pipeline(dictionary, model=lda)
#     topic_data = pipeline.prepare_topic_data(corpus)
#     app = topicwizard.get_dash_app(topic_data)
