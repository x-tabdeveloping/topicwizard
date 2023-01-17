Getting Started
==================

Installation
^^^^^^^^^^^^

topicwizard can be simply installed by installing the PyPI package.

.. code-block::

   pip install tweetopic

Usage
^^^^^^^^^
Train a scikit-learn compatible topic model.

.. code-block:: python

   from sklearn.decomposition import NMF
   from sklearn.feature_extraction.text import CountVectorizer
   from sklearn.pipeline import Pipeline

   bow_vectorizer = CountVectorizer()
   nmf = NMF(n_components=10)
   topic_pipeline = Pipeline(
      [
         ("bow", bow_vectorizer),
         ("nmf", nmf),
      ]
   )
   topic_pipeline.fit(texts)

Visualize with topicwizard.

.. code-block:: python

   import topicwizard

   topicwizard.visualize(pipeline=topic_pipeline, corpus=texts)

   # You can also specify vectorizer and topic_model separately
   topicwizard.visualize(vectorizer=bow_vectorizer, topic_model=nmf, corpus=texts)

   # You can pass in names of documents if you have a named corpus
   topicwizard.visualize(pipeline=topic_pipeline, corpus=texts, document_names=document_names)

   # If you already have named topics, you can also pass in topic names
   topicwizard.visualize(pipeline=topic_pipeline, corpus=texts, topic_names=topic_names)

This will open a new browser tab in which you can investigate topic models visually.
:ref:`Topics <usage topics>`

:ref:`Documents <usage documents>`

:ref:`Words <usage words>`