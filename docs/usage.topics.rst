.. _usage topics:

Investigating Topics
=======================

Web App
-----------

When using the app and clicking 'Topics' on the navigation bar you will be presented with this screen.

.. image:: _static/screenshot_topics.png
    :width: 800
    :alt: Screenshot of topics.


This page intends to be a drop in replacement for PyLDAvis, and displays word importances
and the distances and relative sizes of topics.
All plots in the app can be saved by clicking on the small camera icon on the plot in the top right.

If you only intend to display this page, you can disable the others by specifying them when calling visualize().

.. code-block:: python

   topicwizard.visualize(corpus=texts, pipeline=pipeline, exclude_pages=["documents", "words"])

Topic map
^^^^^^^^^^

On the left you will see a plot showing you all the topics, aka. the topic map.

.. image:: _static/topics_topic_map.png
    :width: 800
    :alt: Topic map.

Topic positions are calculated from model parameters (the topic-term matrix) with UMAP.
The graph is draggable with the cursor and zoomable by scrolling.
To select a topic click on it.

Wordcloud and Barplot
^^^^^^^^^^^^^^^^^^^^^^
The most relevant words for the given topic are displayed on the right in the form of a
bar chart and a wordcloud.

.. image:: _static/topic_bar_wordcloud.png
    :width: 800
    :alt: Barchart and wordcloud.

The wordcloud is draggable with the cursor and zoomable by scrolling.

Relevance
^^^^^^^^^^^

Word relevance for a given topic is calculated using the relevance metric in the LDAvis paper.
You can intuitively think of it as a way to specify how topic-specific you want the appearing words to be,
with o% representing highly topic-specific, and 100% representing not topic-specific.
You can adjust the relevance metrix (lambda) by using this slider.

.. image:: _static/topic_slider.png
    :width: 800
    :alt: Relevance slider.

Rename Topics
^^^^^^^^^^^^^^
You can rename topics by clicking the textfield on the top and starting to type.

.. image:: _static/topic_renamer.png
    :width: 800
    :alt: Topic renamer.

Self-Contained Plots
--------------------

It might be an overkill for you to display the entire page, and you might want static html plots instead of the entire application running.
This can be particulary useful for reports with DataPane or Jupyter Notebooks.

Topic Map
^^^^^^^^^
You can display the same topic map as with the app.

.. code-block:: python
   
   from topicwizard.figures import topic_map

   topic_map(corpus=texts, pipeline=pipeline)


.. raw:: html
   :file: _static/plots/topic_map.html

Word Barplots
^^^^^^^^^^^^^

You can display a joint plot of all topics, where word importances are displayed on a bar chart.
You can specify the relevance metric with the alpha keyword parameter.

.. code-block:: python
   
   from topicwizard.figures import topic_barcharts

   topic_barcharts(corpus=texts, pipeline=pipeline, alpha=1.0)

If you find that too many words get displayed, you can reduce that with the top_n keyword.

.. code-block:: python

   topic_barcharts(corpus=texts, pipeline=pipeline, top_n=5)

.. raw:: html
   :file: _static/plots/topic_barcharts.html

Word Clouds
^^^^^^^^^^^^^

You can produce a joint word cloud plot of all topics.
You can specify the relevance metric with the alpha keyword parameter.

.. code-block:: python

   from topicwizard.figures import topic_wordclouds

   topic_wordclouds(corpus=texts, pipeline=pipeline, alpha=1.0)


.. raw:: html
   :file: _static/plots/topic_wordclouds.html
