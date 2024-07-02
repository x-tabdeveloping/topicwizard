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

Easy Deployment (New in version 1.1.0)
--------------------------------------

If you want to produce a deployment of topicwizard with a fitted topic model, you can now produce a Docker deployment folder with easy_deploy().

.. code-block:: python

    import joblib
    import topicwizard

    # Load previously produced topic_data object
    topic_data = joblib.load("topic_data.joblib")

    topicwizard.easy_deploy(topic_data, dest_dir="deployment", port=7860)

    # deployment/
    #    - Dockerfile
    #    - main.py
    #    - topic_data.joblib


This will put everything you need in the `deployment/` directory, and will work out of the box on cloud platforms or HuggingFace Spaces.


Cold starts are now faster, as UMAP projections can be precomputed.

.. code-block:: python

    topic_data_w_positions = topicwizard.precompute_positions(topic_data)


Deploying to HuggingFace Spaces
-------------------------------

You can deploy topic models in topicwizard to HuggingFace Spaces by creating a `Docker Space <https://huggingface.co/docs/hub/spaces-sdks-docker>`_.
You should then clone the repository to your computer.

.. code-block:: bash

   git clone <link_to_space>

Then move the contents of a deployment folder created with easy_deploy() to the repo folder, and push everything to the Space.

.. code-block:: bash

   mv deployment/* /path/to/space_repo
   cd path/to/space_repo
   git add -A
   git commit -m "Added deployment"
   git push

Your deployment will automatically start.
