.. _usage pipelines:

Pipelines
=============

topicwizard uses the idea of a topic pipeline as its main abstraction to understand topic models.
Most of topicwizard's code relies on the assumption that any topic model consists of a pipeline
of a vectorizer component, and some sort of a decomposition model.
One can either use a regular sklearn Pipeline, or topicwizards own abstraction, TopicPipeline.


Vectorizer
----------
The vectorizer is a component that turns texts into bag-of-words vectors.
A sensible default would be scikit-learn's CountVectorizer, which
makes this process rather customizable and is quite reliable.

.. code-block:: python

   from sklearn.feature_extraction.text import CountVectorizer

   vectorizer = CountVectorizer()


Topic Model
------------
topicwizard assumes that topic models are some sort of decomposition model,
that can turn bag-of-words, or similar representations into a decomposed signal of topics.
Anything that turns a document-term-matrix into a document-topic matrix is considered a topic model
by topicwizard.
We additionally require that all models have to have a ".components_" attribute,
which is a topic-term importance matrix.
Good examples of of this are Non-negative Matrix Factorization, or Latent Dirichlet Allocation from scikit-learn.
Some thrid-party libraries, such as tweetopic also come with sklearn-compatible components.

.. code-block:: python

   # LDA for long texts
   from sklearn.decomposition import LatentDirichletAllocation

   model = LatentDirichletAllocation(n_components=10)

   # You can use NMF too
   from sklearn.decomposition import NMF

   model = NMF(n_components=10)

   # Or tweetopic's DMM for short texts
   # pip install tweetopic

   from tweetopic import DMM

   model = DMM(n_components=10)

Pipeline
--------
You can string these components together into a pipeline, and can even add additional transformations in the middle.

.. code-block:: python

   from sklearn.pipeline import make_pipeline

   topic_pipeline = make_pipeline(vectorizer, model)

.. image:: _static/pipeline.png
    :width: 400
    :alt: Schematic overview of pipeline.

TopicPipeline
-------------
TopicPipeline is a subclass of scikit-learn Pipelines, and for the most part is functionally identical to using
a regular Pipeline. We recommend that you use TopicPipeline instead of a regular pipeline as it is more convenient to
use in downstream tasks and model interpretation.

.. code-block:: python

   from topicwizard.pipeline import make_topic_pipeline

   topic_pipeline = make_topic_pipeline(vectorizer, model)

Named Outputs
^^^^^^^^^^^^^^^^^^^^
Topic Pipelines do automatic topic name inference upon fitting, this can be useful if you intend to
use these names further down a pipeline for example:

.. code-block:: python

   topic_pipeline.fit(texts)
   print(topic_pipeline.get_feature_names_out())

Freezing Components
^^^^^^^^^^^^^^^^^^^^
If you intend to use the topics in a pipeline downstream, you might want to first train a topic model,
interpret the topics with topicwizard, and then train downstream components.
In these cases you can freeze the vectorizer and the topic model, so that they do not change when you call fit() or partial_fit()
on an outer pipeline.

.. code-block:: python

   from sklearn.pipeline import make_pipeline
   from sklearn.linear_model import LogisticRegression

   topic_pipeline = make_topic_pipeline(vectorizer, model).fit(texts)

   # Investigate topics
   topicwizard.visualize(topic_pipeline)

   # Freezing topic pipeline
   topic_pipeline.freeze = True
   # Constructing classification pipeline
   cls_pipeline = make_pipeline(topic_pipeline, LogisticRegression())
   cls_pipeline.fit(X, y)


Output as DataFrame
^^^^^^^^^^^^^^^^^^^^
Scikit-learn pipelines and components can now output pandas DataFrames instead of matrices when asked to.
The issue is that vectorizers do not play very well with this dynamic, since they have sparse outputs,
and pandas cannot deal with sparse matrices.

TopicPipeline allows you to set DataFrames to be the output of your topic pipeline either by
providing a parameter or by using the set_output API in scikit-learn.

.. code-block:: python

   # Set a parameter
   pipeline = make_topic_pipeline(vectorizer, model, pandas_out=True)

   # Or use set_output API
   pipeline = make_topic_pipeline(vectorizer, model).set_output(transform="pandas")

This is insanely useful when you are trying to investigate the topic content of individual documents.
You can for example display a heatmap of topics in a set of documents as such:

.. code-block:: python

   import plotly.express as px

   texts = [
      "Coronavirus killed 50000 people today.",
      "Donald Trump's presidential campaing is going very well",
      "Protests against police brutality have been going on all around the US.",
   ]
   topic_df = pipeline.transform(texts)
   topic_df.index = texts
   px.imshow(topic_df).show()

.. raw:: html
   :file: _static/plots/document_topic_heatmap.html

Alternatively you can use human-learn to create rule based components around your topic model.

Here's an example of how you could construct a classification pipeline for seeing which document
is about Covid using a topic model we train and investigate.
These kind of pipelines can be very useful when you do not have labelled data but would still like to
filter or label texts.

.. code-block:: python

   # Install human-learn from PyPI 
   # pip install human-learn

   from hulearn.classification import FunctionClassifier
   from sklearn.pipeline import make_pipeline

   topic_pipeline = make_topic_pipeline(vectorizer, model).fit(texts)

   # Investigate topics
   topicwizard.visualize(topic_pipeline)

   # Creating rule for classifying something as a corona document
   def corona_rule(df, threshold=0.5):
       is_about_corona = df["11_vaccine_pandemic_virus_coronavirus"] > threshold
       return is_about_corona.astype(int)
   
   # Freezing topic pipeline
   topic_pipeline.freeze = True
   classifier = FunctionClassifier(corona_rule)
   cls_pipeline = make_pipeline(topic_pipeline, classifier)

Pseudoprobabilites
^^^^^^^^^^^^^^^^^^^^
TopicPipeline can be instructed to normalize document-topic importances as if they were probabilites.
This is useful if you want to treat importances as probabilities in calculations, or when specifying thresholds.


.. code-block:: python

   pipeline = make_topic_pipeline(vectorizer, model, norm_row=True)
   # Or set it to false if you want to turn it off
   pipeline = make_topic_pipeline(vectorizer, model, norm_row=False)

Validation
^^^^^^^^^^^^^^^^^^^^
TopicPipeline validates whether the passed components are appropriate to use as a topic model in topicwizard
unlike a regular scikit-learn pipeline.

