.. _usage words:

Investigating Words
=======================

You can investigate relations of words with other words or with topics in topicwizard.

The fundamental trick is that you can interpret topic-word importances/probabilities as word
embeddings, and this is in fact what people do who train LSI word embeddings.

For example:

.. code-block:: python

   # SVD is just the underlying mathematical algorithm for Latent Semantic Indexing/Allocation
   from sklearn.decomposition import TruncatedSVD
   from sklearn.feature_extraction.text import CountVectorizer
   from sklearn.pipeline import make_pipeline

   vectorizer = CountVectorizer()
   # Word embeddings with 100 dimensions
   svd = TruncatedSVD(n_components=100)
   pipeline = make_pipeline(vectorizer, svd)
   pipeline.fit(corpus)

   # Your word embeddings are the components of the topic model transformed
   # You can virtually use any topic model like this.
   embeddings = svd.components_.T
   # THis is the order of words in the embedding matrix
   vocab = vectorizer.get_feature_names_out()


This means that you can also use topicwizard for investigating certain word embedding models.

Web App
-----------

When you click 'Words' on the navigation bar of the web application you will be presented with this screen.

.. image:: _static/screenshot_words.png
    :width: 800
    :alt: Screenshot of words.

This is mainly dedicated to interpreting word associations and their relations to topics.
If you only want to see this page you should disable all else when calling visualize().


.. code-block:: python

   topicwizard.visualize(corpus=texts, pipeline=pipeline, exclude_pages=["topics", "documents"])


To get more information about how to interpret the graphs and how they are produced, hover your cursor over the helper in the corner.

Self-Contained Plots
--------------------

It might be an overkill for you to display the entire page, and you might want static html plots instead of the entire application running.
This can be particulary useful for reports with DataPane or Jupyter Notebooks.

Word Map
^^^^^^^^^

The word map that you can display with a dedicated function is slightly different from the one in the app
as here you can't select words to highlight.

Instead you can specify a cutoff in Z-values over which words will be labelled on the graph.

Words are also distinctively colored according to the most relevant topic as you cannot select
the individual words for inspection.

.. code-block:: python
   
   from topicwizard.figures import word_map

   word_map(corpus=texts, pipeline=pipeline)


.. raw:: html
   :file: _static/plots/word_map.html

Topics
^^^^^^^^^

You can visualize most relevant topics for a given set of words with
barcharts, these behave virtually the same as in the app, but no associations
are selected by default.

So for example if we would like to know which topics contain the words "supreme" and "court", we can
do so:

.. code-block:: python
   
   from topicwizard.figures import word_association_barchart

   word_association_barchart(["supreme", "court"], corpus=texts, pipeline=pipeline)


.. raw:: html
   :file: _static/plots/word_association_barchart.html

