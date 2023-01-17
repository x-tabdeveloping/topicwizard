.. _usage persistence:

Persistence
=============

You can persist your results by clicking on the download button right next to the navigation bar.

.. image:: _static/download_button.png
    :width: 200
    :alt: Download button.

This will serialize all the necessary data for you to save the topic model and the topic names into a joblib file.

Then you can load this data in another script and start the app from the persisted data.

.. code-block:: python

    import topicwizard

    topicwizard.load(filename="topic_data.joblib")