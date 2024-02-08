Deployment
============

Since topicwizard is technically just like any Dash application you can easily deploy topicwizard
to a cloud provider or your own servers.

If you have access to a TopicData object, you can build a Dash application, that can be used to spin up a server.

.. code-block:: python

    # main.py
    import topicwizard

    app = topicwizard.get_dash_app(topic_data)

Then you can run the server from a main file manually:

.. code-block:: python
    
    # main.py
    if __name__ == "__main__":
        app.run_server(debug=False, port=8050)

We recommend using `Gunicorn <https://gunicorn.org/>`_ for deployment in production.

.. code-block::

    gunicorn main:app.server -b 8050

You can easily package a topicwizard app with gunicorn into a Docker image as well.

