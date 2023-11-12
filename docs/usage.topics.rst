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

To get more information about how to interpret the graphs and how they are produced, hover your cursor over the helper in the corner.

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
