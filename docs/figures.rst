.. _figures:

Individual Figures
==================

If you are preparing individual figures for a publication or report,
you might need to modify the appearance or resolution of the figures you producing.

It might also be the case that it's only certain figures you're interested in.

topicwizard comes with an interface that allows you to do just that.
If you have a TopicData object, you can manually produce individual figures.

These figures are like any other interactive Plotly figure, therefore you can manipulate them as such,
and export them as HTML or a number of image formats.
For an extensive overview of how you can manipulate plots produced by topicwizard consult `Plotly's documentation <https://plotly.com/python/>`_.

Topic Map
^^^^^^^^^

You can display a semantic map of topics in your model.

.. code-block:: python
   
   from topicwizard.figures import topic_map

   topic_map(topic_data)


.. raw:: html

    <iframe src="_static/plots/topic_map.html" width="800px" height="600px"></iframe>

.. autofunction:: topicwizard.figures.topic_map

Word Barplots
^^^^^^^^^^^^^

You can display a joint plot of all topics, where word importances are displayed on a bar chart.
You can specify the relevance metric with the alpha keyword parameter.

.. code-block:: python
   
   from topicwizard.figures import topic_barcharts

   topic_barcharts(topic_data)

If you find that too many words get displayed, you can reduce that with the top_n keyword.

.. code-block:: python

   topic_barcharts(topic_data, top_n=5)

.. raw:: html

    <iframe src="_static/plots/topic_barcharts.html" width="800px" height="600px"></iframe>

.. autofunction:: topicwizard.figures.topic_barcharts

Word Clouds
^^^^^^^^^^^^^

You can produce a joint word cloud plot of all topics.
You can specify the relevance metric with the alpha keyword parameter.

.. code-block:: python

   from topicwizard.figures import topic_wordclouds

   topic_wordclouds(topic_data)


.. raw:: html

   <iframe src="_static/plots/topic_wordclouds.html" width="800px" height="600px"></iframe>

.. autofunction:: topicwizard.figures.topic_wordclouds


Word Map
^^^^^^^^^

The word map that you can display with a dedicated function is slightly different from the one in the app
as here you can't select words to highlight.

Instead you can specify a cutoff in Z-values over which words will be labelled on the graph.

Words are also distinctively colored according to the most relevant topic as you cannot select
the individual words for inspection.

You can either choose to let UMAP discover the axis and project the words into 2D space, which is good for exploring words' distances and relations to each other in the model,
as well as potential clusters of words in the topic model.

.. code-block:: python
   
   from topicwizard.figures import word_map

   word_map(topic_data)

.. raw:: html

   <iframe src="_static/plots/word_map.html" width="800px" height="600px"></iframe>

Or you can display words with given topics as axes. This is especially useful for models like Semantic Signal Separation or Latent Semantic Analysis,
where words with the lowest importance for a topic also cary information, as a topic is assumed to be an axis of semantic space.

.. code-block:: python
   
   from topicwizard.figures import word_map

   word_map(
     topic_data,
     topic_axes=(
        "9_api_apis_register_automatedsarcasmgenerator",
        "4_study_studying_assessments_exams"
     )
   )

.. raw:: html

   <iframe src="_static/plots/word_map_axes.html" width="800px" height="600px"></iframe>


.. autofunction:: topicwizard.figures.word_map

Important Topics
^^^^^^^^^^^^^^^^

You can visualize most relevant topics for a given set of words with
barcharts, these behave virtually the same as in the app, but no associations
are selected by default.

So for example if we would like to know which topics contain the words "supreme" and "court", we can
do so:

.. code-block:: python
   
   from topicwizard.figures import word_association_barchart

   word_association_barchart(topic_data, ["supreme", "court"])


.. raw:: html

   <iframe src="_static/plots/word_association_barchart.html" width="800px" height="600px"></iframe>

.. autofunction:: topicwizard.figures.word_association_barchart


Document Map
^^^^^^^^^^^^

You can display a map of documents as a self-contained plot.
This can be advantageous when you want to see how different documents relate to each other in your corpus,
and to the underlying topics discovered by the model. 

This plot is not entirely identical to the one in the app, as documents cannot be selected or searched for.

Different topics are clearly outlined with discrete colors.

.. code-block:: python
   
   from topicwizard.figures import document_map

   document_map(topic_data)


.. raw:: html

   <iframe src="_static/plots/document_map.html" width="800px" height="600px"></iframe>

.. autofunction:: topicwizard.figures.document_map

Topic Distribution
^^^^^^^^^^^^^^^^^^

You can display topic distributions for a given document or list of documents on a bar chart.

.. code-block:: python

   from topicwizard.figures import document_topic_distribution

   document_topic_distribution(
       topic_data,
       "New cure against type 2 diabetes in development.",
   )

.. raw:: html

   <iframe src="_static/plots/document_topic_distribution.html" width="800px" height="600px"></iframe>

.. autofunction:: topicwizard.figures.document_topic_distribution

You can also display topic distribution over time in a single document on a line chart.
(or an entire corpus if you join the texts.)

This works by taking windows of tokens from the document and running them through the pipeline.
You can specify window and step size in number of tokens if you find the results have to high or to low resolution.


.. code-block:: python

   from topicwizard.figures import document_topic_timeline

   document_topic_timeline(
       topic_data,
       "New cure against type 2 diabetes in development.",
   )

.. raw:: html

   <iframe src="_static/plots/document_topic_timeline.html" width="800px" height="600px"></iframe>

.. autofunction:: topicwizard.figures.document_topic_timeline


Group Map
^^^^^^^^^

You can display the group map as a standalone plot, with the groups being colored according to dominant topic.

.. code-block:: python
   
   from topicwizard.figures import group_map

   group_map(topic_data, group_labels)


.. raw:: html

   <iframe src="_static/plots/group_map.html" width="800px" height="600px"></iframe>

.. autofunction:: topicwizard.figures.group_map

Group Topic Barcharts
^^^^^^^^^^^^^^^^^^^^^

You can create a joint plot of the topic content of all groups.
These will be displayed as bar charts.

.. code-block:: python
   
   from topicwizard.figures import group_topic_barcharts

   group_topic_barcharts(corpus, group_labels, pipeline=pipeline, top_n=5)

.. raw:: html

   <iframe src="_static/plots/group_topic_barcharts.html" width="800px" height="600px"></iframe>

.. autofunction:: topicwizard.figures.group_topic_barcharts

Group Word Clouds
^^^^^^^^^^^^^^^^^

You can create word clouds for each of the group labels. This will only take word counts into account and not relevance.

.. code-block:: python

   from topicwizard.figures import group_wordclouds

   group_wordclouds(corpus, group_labels, pipeline=pipeline)


.. raw:: html

   <iframe src="_static/plots/group_wordclouds.html" width="800px" height="600px"></iframe>

.. autofunction:: topicwizard.figures.group_wordclouds
