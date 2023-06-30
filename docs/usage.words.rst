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


Word map
^^^^^^^^^^^^^

On the left you will see a plot showing you all the words, aka. the word map.
On the map positions are determined by 2-dimensional UMAP projections of the transposed
components of the topic model.

.. image:: _static/word_map.png
    :width: 800
    :alt: Word map.

Selecting Words
^^^^^^^^^^^^^^^^^^^^

You can select words by typing them into the field on the top left and searching for them.
Multiple words may be selected at the same time.
Clicking a word on the map adds the word to the selection.

Associations
^^^^^^^^^^^^^
Closely associated words also get highlighted on the graph and are included in calculations.
The most closely associated words are the ones that have the lowest distance to the given words.
Topicwizard finds the specified amount of closest words to the selected ones.
The number of associated words can be adjusted with this slider:

.. image:: _static/association_slider.png
    :width: 800
    :alt: Association slider.

Important Topics
^^^^^^^^^^^^^^^^^^^^^^
You can see which topics use the selected word and their associations most frequently by glancing at the bar plot
to the right.

.. image:: _static/word_barplot.png
    :width: 800
    :alt: Bar chart.

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

