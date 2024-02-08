.. _usage persistence:

Persistence
=============

You can persist your results by clicking on the download button right next to the navigation bar.

.. image:: _static/download_button.png
    :width: 200
    :alt: Download button.

This will serialize all inference information into a `TopicData <topic data>`_ object,
including the topic names you have specified while using the application.

This is virtually equivalent to saving the TopicData on the server directly,
using the prepare_topic_data() method of compatible models or `TopicPipelines <usage pipelines>`_,
except for manually assigned topic names.

Serialization is done with `joblib <https://joblib.readthedocs.io/en/stable/>`_.

.. code-block:: python

   from turftopic import KeyNMF
   import joblib

   model = KeyNMF(10)
   topic_data = model.prepare_topic_data(corpus)

   joblib.dump(topic_data, "topic_data.joblib")


Then this data can be loaded again using joblib.

.. code-block:: python

   import topicwizard
   # We import this only for type checking
   from topicwizard.data import TopicData
   import joblib

   topic_data: TopicData = joblib.load("topic_data.joblib")

   topicwizard.visualize(topic_data=topic_data)

