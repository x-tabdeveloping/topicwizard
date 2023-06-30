Web Application
==================

As stated earlier topicwizard is easiest to use with the default visualization dashboard that comes with it,
as this provides a general and interactive overview of the topic models you wish to interpret.

For this example let's train a Non-negative Matrix Factorization model over a corpus of texts we have.

.. code-block:: python

   # Training a compatible topic model
   from sklearn.decomposition import NMF
   from sklearn.feature_extraction.text import CountVectorizer
   from sklearn.pipeline import make_pipeline

   bow_vectorizer = CountVectorizer()
   nmf = NMF(n_components=10)
   pipeline = make_pipeline(bow_vectorizer, nmf)
   topic_pipeline.fit(texts)

Once you have trained a scikit-learn compatible topicmodel, like NMF, 
interpreting the model in topicwizard is as trivial as starting the web application with the visualize() function.

Topicwizard can either take a pipeline, where the first element is a CountVectorizer (or functionally identical)
and the last element is a topic model...

.. code-block:: python

   import topicwizard

   topicwizard.visualize(corpus=texts, pipeline=pipeline)

Or it can also take these components in the pipeline individually.

.. code-block:: python

   topicwizard.visualize(corpus=texts, vectorizer=bow_vectorizer, topic_model=topic_model)


.. image:: _static/screenshot_topics.png
    :width: 800
    :alt: Screenshot of topics.

This will open a web app in a new browser tab with three pages, one for topics, one for words and one for documents,
where you can investigate the intricate relations of these in an interactive fashion.

Beware that if you display all three pages, especially with larger corpora or vocabularies, topicwizard might take a long time
to start up.
This is because visualizing documents, words and topics is hard. You need to have 2D projections of their embeddings, for
which topicwizard uses a method called UMAP, which produces nicely interpretable projections, but takes a long time to train
and infer.

If you just want a drop-in replacement of PyLDAvis for your project, and you only care about word importances for
your topics, you can disable the words and documents pages with the exclude_pages argument.


.. code-block:: python

   topicwizard.visualize(corpus=texts, pipeline=pipeline, exclude_pages=["documents", "words"])

Or equally if you use a matrix decomposition method for creating word embeddings like LSI for example, you can use topicwizard to visualize your embeddings
and disable all else:

.. code-block:: python

   topicwizard.visualize(corpus=texts, pipeline=pipeline, exclude_pages=["documents", "topics"])

