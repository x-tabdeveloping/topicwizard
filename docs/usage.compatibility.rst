.. _usage compatibility:

Compatibility
==============

Compatibility layers have been introduced with version 0.2.4,
Gensim and BERTopic can now be used with topicwizard.

Gensim
^^^^^^

sklearn compatible pipelines for Gensim models can be created once you have a dictionary
and a topic model object available, topicwizard is compatible with LSI, LDA (also multicore) and NMF.

First you need to train a Gensim dictionary and topic model.

.. code-block:: python

  from gensim.corpora.dictionary import Dictionary
  from gensim.models import LdaModel

  texts: list[list[str] = [
      ['computer', 'time', 'graph'],
      ['survey', 'response', 'eps'],
      ['human', 'system', 'computer'],
      ...
  ]
  dictionary = Dictionary(texts)
  bow_corpus = [dictionary.doc2bow(text) for text in texts]
  lda = LdaModel(bow_corpus, num_topics=10)

Then you need to create a pipeline with topicwizard.

.. code-block:: python

   import topicwizard

   pipeline = topicwizard.gensim_pipeline(dictionary, model=lda)
   # Then you can use the pipeline as usual
   corpus = [" ".join(text) for text in texts]
   topicwizard.visualize(pipeline=pipeline, corpus=corpus)

BERTopic
^^^^^^^^

You can create a topicwizard pipeline from a BERTopic pipeline fairly easily.

First you need to train a BERTopic topic model.

.. code-block:: python

    from bertopic import BERTopic

    model = BERTopic(corpus)

Then you need to create a pipeline with topicwizard.

.. code-block:: python

   import topicwizard

   # BERTopic automatically assigns topic names, you can use these
   # in topicwizard
   pipeline, topic_names = topicwizard.bertopic_pipeline(model)

   # Then you can use the pipeline as usual
   topicwizard.visualize(pipeline=pipeline, corpus=corpus)

.. note::
   BERTopic compatibility is an experimental feature in topicwizard.
   Most of topicwizard rests on the bag of words assumption, and two-step topic
   pipelines, which BERTopic does not conform to.
   Document and word positions for example are solely based on c-TF-IDF representations,
   not on the contextual embeddings in BERTopic.

   If you find that the results are unsatisfactory, we recommend that you use BERTopic's
   own excellent visualizations. (They are honestly pretty great :))
   In the future there is a possiblity of a BERTopic-specific visualization dashboard.


Custom Topic Models
^^^^^^^^^^^^^^^^^^^^^^^^

You can write topic models, which are compatible with topicwizard.
If you have a topic model, which rests on the bag-of-words assumption this is
a fairly straightforward task.

Vectorizer components of the pipeline should have the following properties:

.. code-block:: python

   from typing import Iterable
   
   import numpy as np
   from sklearn.base import BaseEstimator

   # All of your components should ideally be inherited from BaseEstimator
   class CustomVectorizer(BaseEstimator):
   
      # All vectorizers should have a transform method,
      # that turns raw texts into sparse arrays 
      # of shape (n_documents, n_features)
      def transform(self, raw_documents: Iterable[str], y=None):
          pass

      # All vectorizers should have a get_feature_names_out method, that
      # returns a dense array of feature names
      def get_feature_names_out(self) -> np.ndarray:
          pass

Topic model components should follow the following structure:

.. code-block:: python

   # Same thing, BaseEstimator is a good thing to have
   class CustomTopicModel(BaseEstimator):

       # All topic models should have a transform method, that takes
       # the vectorized documents and returns a sparse or dense array of
       # topic distributions with shape (n_docs, n_topics)
       def transform(self, X):
           pass

       # All topic models should have a property or attribute named
       # components_, that should be a dense or sparse array of topic-word
       # distributions of shape (n_topics, n_features)
       @property
       def components_(self) -> np.ndarray:
           pass

