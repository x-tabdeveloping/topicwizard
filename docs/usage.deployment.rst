Deployment
============

Since topicwizard is technically just like any Dash application you can easily deploy topicwizard
to a cloud probvider or your own servers.

You can either create the app from scratch with a trained topic pipeline.

.. code-block:: python

    # main.py
    import topicwizard

    app = topicwizard.get_dash_app(vectorizer, topic_model, corpus=corpus)

Or you can retrieve an app from loaded data.

.. code-block:: python

    # main.py
    app = topicwizard.load_app(filename="topic_data.joblib")

Then you can run the server from the file manually:

.. code-block:: python
    
    # main.py
    if __name__ == "__main__":
        app.run_server(debug=False, port=8050)

We recommend using `Gunicorn <https://gunicorn.org/>`_ for deployment in production.

.. code-block::

    gunicorn main:app.server -b 8050

You can easily package a topicwizard app with gunicorn into a Docker image as well.

