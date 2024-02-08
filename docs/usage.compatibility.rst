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
   from topicwizard.compatibility import gensim_pipeline

   pipeline = gensim_pipeline(dictionary, model=lda)
   # Then you can use the pipeline as usual
   corpus = [" ".join(text) for text in texts]
   topicwizard.visualize(pipeline=pipeline, corpus=corpus)

.. autofunction:: topicwizard.compatibility.gensim.gensim_pipeline

BERTopic
^^^^^^^^

You can create a topicwizard pipeline from a BERTopic pipeline fairly easily.

First you need to construct a BERTopic model.
The model does not have to be pretrained. If you don't train it, it will be automatically fitted when running the app.

BERTopic models have to be wrapped in a compatibility layer to be used with topicwizard.

.. note::

   BERTopic models are now first-class citizens of topicwizard, and have native support.

.. code-block:: python

   from bertopic import BERTopic
   from topicwizard.compatibility import BERTopicWrapper

   model = BERTopic(language="english")
   wrapped_model = BERTopicWrapper(model)

You can either produce a TopicData object with this model or use it directly in the web app.

.. code-block:: python
   
   import topicwizard

   # Start the web app immediately
   topicwizard.visualize(corpus, model=wrapped_model)
   
   # Or produce a TopicData object for persistance or figures.
   topic_data = wrapped_model.prepare_topic_data(corpus)

.. autoclass:: topicwizard.compatibility.bertopic.BERTopicWrapper
   :members:

Custom Topic Models
^^^^^^^^^^^^^^^^^^^^^^^^

Classical Models with TopicPipeline
-----------------------------------

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


Any Model / Contextual Models
-----------------------------

Contextual models have to follow the following interface, and have to be able to produce a TopicData objects:

.. code-block:: python

   from topicwizard.model_interface import TopicModel
   from topicwizard.data import TopicData

   # TopicModel is only a Protocol, the model inferits no behaviour,
   # it just provides static checks
   class CustomTopicModel(TopicModel):
      def prepare_topic_data(
          self,
          corpus: list[str],
      ) -> TopicData:
          pass

