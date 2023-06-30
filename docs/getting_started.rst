Getting Started
==================

Installation
^^^^^^^^^^^^

topicwizard can be simply installed by installing the PyPI package.

.. code-block::

   pip install topic-wizard

Usage
^^^^^^^^^
.. raw:: html

   <a href="https://colab.research.google.com/github/x-tabdeveloping/topic-wizard/blob/main/examples/basic_usage.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


Train a scikit-learn compatible topic model.

.. note::
   If you intend to investigate non-scikit-learn models, please have a look at
   :ref:`Compatibility <usage compatibility>`


.. code-block:: python

   from sklearn.decomposition import NMF
   from sklearn.feature_extraction.text import CountVectorizer
   from sklearn.pipeline import make_pipeline

   bow_vectorizer = CountVectorizer()
   nmf = NMF(n_components=10)
   pipeline = make_pipeline(bow_vectorizer, nmf)
   topic_pipeline.fit(texts)

The easiest and most sensible way to visualize is with the topicwizard web application.

.. code-block:: python

   import topicwizard

   topicwizard.visualize(pipeline=topic_pipeline, corpus=texts)


This will open a new browser tab in which you can investigate topic models visually.
