.. _topic data:

Topic Data
^^^^^^^^^^

Topic data is the main abstraction in topicwizard that contains information about topical inference in a corpus,
that can be used to reproduce the interpretive visualizations in the web app and individual figures.

This interface is needed so that topicwizard can use data from other libraries or your own topic models,
and inference data can be persisted and used across different machines.

The TopicData type is what's referred to as a TypedDict in Python.
What this means is that TopicData is essentially just a dictionary at runtime,
and is as such interoperable with anything else in Python,
but static type checking is provided if you are using a type checker in your editor, like Pyright.

All visualization utils at least optionally take this object.
This means that if you have a TopicData object from some corpus and some topic model, you can reproduce all visualizations using this object.

.. code-block:: python
   
   import topicwizard
   from topicwizard.figures import topic_map

   # Usage with figures
   topic_map(topic_data)

   # Usage with web app
   # Beware that topic_data is a keyword argument
   topicwizard.visualize(topic_data=topic_data)


.. autoclass:: topicwizard.data.TopicData
   :members:

