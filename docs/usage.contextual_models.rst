.. _usage contextual:

Contextually Sensitive Topic Models
===================================

Modern topic models, and state of the art approaches no longer rely on the bag-of-words assumption,
and are sensitive to contextual nuances.

topicwizard hass native support for contextual models from the `Turftopic <https://github.com/x-tabdeveloping/turftopic>`_ Python package, and has utilites to be used with `BERTopic <https://github.com/MaartenGr/BERTopic>`_.

We opted not to implement wrappers for Top2Vec and CTM as most of their functionality can be achieved by building clustering or autoencoding topic models in Turftopic,
and they are not as popular in research and the industry as BERTopic.

For an extensive overview of models, tutorials, and theoretical background on contextually sensitive topic models consult the `Turftopic package's documentation <https://x-tabdeveloping.github.io/turftopic/model_overview/>`_.

Example
----------
The following example demonstrates how to interpret contextual models using Turftopic.
We will build a Semantic Signal Separation model, and visualize it with the topicwizard web application.

.. code-block:: python

    import topicwizard
    from turftopic import SemanticSignalSeparation

    model = SemanticSignalSeparation(n_components=10)

    # You can produce the topic data from a corpus before running the app
    # This option should be prefered as the data can be saved and the app can be restarted
    # Or you can use it for producing individual figures later.
    topic_data = model.prepare_topic_data(corpus)
    topicwizard.visualize(topic_data=topic_data)

    # Or you can run the app directly with the model and a corpus
    topicwizard.visualize(corpus, model=model)

BERTopic models have to be wrapped in a compatibility layer to be used:


.. code-block:: python

    from bertopic import BERTopic
    from topicwizard.compatibility import BERTopicWrapper

    # The model can be fitted or not.
    model = BERTopic()
    wrapped_model = BERTopicWrapper(model)

    topicwizard.visualize(corpus, model=wrapped_model)
