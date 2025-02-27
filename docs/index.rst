Getting Started
==================

topicwizard is a pretty and opinionated Python library for topic model visualization built on Dash and Plotly.
This website contains the user guide to topicwizard as well as the API reference.

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


Build a scikit-learn compatible topic pipeline with classical models.

.. note::
   If you intend to investigate non-scikit-learn models, please have a look at
   :ref:`Compatibility <usage compatibility>`


.. code-block:: python

   from sklearn.decomposition import NMF
   from sklearn.feature_extraction.text import CountVectorizer
   from topicwizard.pipeline import make_topic_pipeline

   bow_vectorizer = CountVectorizer()
   nmf = NMF(n_components=10)
   model = make_topic_pipeline(bow_vectorizer, nmf)

Or build a contextually sensitive topic model with Turftopic or BERTopic:

.. code-block:: python

   from turftopic import KeyNMF

   model = KeyNMF(n_components=10)

The easiest and most sensible way to visualize is with the topicwizard web application.

.. code-block:: python

   import topicwizard

   topicwizard.visualize(texts, model=model)

.. image:: _static/screenshot_topics.png
    :width: 800
    :alt: Screenshot of topics.

You can also you individual interactive plots to create individual visualizations you might be interested in.

Here is an example of how you can visualize words' relations to each other in a topic model:

.. code-block:: python
   
   from topicwizard.figures import word_map

   topic_data = topic_pipeline.prepare_topic_data(corpus)

   word_map(topic_data)


.. raw:: html
   :file: _static/plots/word_map.html

This will open a new browser tab in which you can investigate topic models visually.

.. toctree::
   :maxdepth: 1
   :caption: Models and Data

   usage.pipelines
   usage.contextual_models
   topic_data
   usage.compatibility

.. toctree::
   :maxdepth: 1
   :caption: Interpretation and Usage

   application
   figures

.. toctree::
   :maxdepth: 1
   :caption: MLOps

   usage.persistence
   usage.deployment


.. toctree::
   GitHub Repository <https://github.com/x-tabdeveloping/topic-wizard>
