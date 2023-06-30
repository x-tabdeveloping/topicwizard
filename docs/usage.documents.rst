.. _usage documents:

Investigating Documents
=======================

Web App
-------

When you click 'Documents' on the navigation bar you will be presented with this page.
This page is mainly for investigating differences between documents and topic distributions in
a corpus.

.. image:: _static/screenshot_documents.png
    :width: 800
    :alt: Screenshot of documents.


.. note::
    With large corpora this page is by far the slowest to start up, as such I would recommend that you disable it
    unless you have special interest in the individual documents in your corpus.
    This is because document representations are high-dimensional and there is usually a lot of documents,
    calculating 2D UMAP projections for such data is slow and tedious.

You can disable the page in visualize()

.. code-block:: python

   topicwizard.visualize(corpus=corpus, pipeline=pipeline, exclude_pages=["documents"])


Document map
^^^^^^^^^^^^^

On the left you will see a plot showing you all the documents, aka. the document map.

.. image:: _static/document_map.png
    :width: 800
    :alt: Document map.

Document positions are calculated from document embeddings created by the vectorizer by reducing dimensionality to 2 with
UMAP.
You can zoom this graph by dragging your cursor and enclosing a selection area.

Selecting documents
^^^^^^^^^^^^^^^^^^^^

You can either select documents by clicking on them on the document map,
or by searching for them in the "Select document" field above the map.

Wordcloud
^^^^^^^^^^^^^^^^^^^^^^
The most used words in the document are displayed on a wordcloud.

.. image:: _static/document_wordcloud.png
    :width: 800
    :alt: wordcloud.

The wordcloud is draggable with the cursor and zoomable by scrolling.

Topic use
^^^^^^^^^^^^^^^^^^^^^^
Use of topics in the document is displayed with a pie chart.

.. image:: _static/document_piechart.png
    :width: 800
    :alt: Pie chart.

Topic Timeline
^^^^^^^^^^^^^^^^^^^^^^
You will also see a timeline which visualizes the use of topics over time in the document.

.. image:: _static/document_timeline.png
    :width: 800
    :alt: Document timeline.

You can remove topics from the plot by clicking them on the right.
You can select individual topics by double clicking on them.

Topic use is calculated with rolling windows of words over the document.
You can adjust window size by dragging the slider on top.

Self-Contained Plots
--------------------

It might be an overkill for you to display the entire page, and you might want static html plots instead of the entire application running.
This can be particulary useful for reports with DataPane or Jupyter Notebooks.

Document Map
^^^^^^^^^^^^

You can display a map of documents as a self-contained plot.
This can be advantageous when you want to see how your topic model maps onto embedding space
or see how different documents relate to each other in the corpus.

This plot is not entirely identical to the one in the app, as documents cannot be selected or searched for.

Different topics are clearly outlined with discrete colors.

You can also choose whether you want to use the representations produced by the vectorizer or the topic model for visualization.
This can be particularly useful if you use a topic model where the representations are not based on the bag-of-words
representations, like BERTopic for example (stay tuned for another fun package btw :)).

.. code-block:: python
   
   from topicwizard.figures import document_map

   # Term-based representations, aka. vectorizer output
   document_map(corpus=texts, pipeline=pipeline, representation="term")


.. raw:: html
   :file: _static/plots/document_map_term.html

.. code-block:: python

   # Topic-based representations, aka. document-topic distributions
   document_map(corpus=texts, pipeline=pipeline, representation="topic")


.. raw:: html
   :file: _static/plots/document_map_topic.html

Topic Distribution
^^^^^^^^^^^^^^^^^^

You can display topic distributions for a given document or list of documents on a pie chart.

.. code-block:: python

   from topicwizard.figures import document_topic_distribution

   document_topic_distribution(
       "Joe Biden takes over presidential office from Donald Trump.",
       pipeline=pipeline,
   )

.. raw:: html
   :file: _static/plots/document_topic_distribution.html

You can also display topic distribution over time in a single document on a line chart.
(or an entire corpus if you join the texts.)

This works by taking windows of tokens from the document and running them through the pipeline.
You can specify window and step size in number of tokens if you find the results have to high or to low resolution.


.. code-block:: python

   from topicwizard.figures import document_topic_timeline

   document_topic_timeline(
       "Joe Biden takes over presidential office from Donald Trump.",
       pipeline=pipeline,
   )

.. raw:: html
   :file: _static/plots/document_topic_timeline.html
