<img align="left" width="82" height="82" src="assets/logo.svg">

# topicwizard

<br>

Pretty and opinionated topic model visualization in Python.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/x-tabdeveloping/topic-wizard/blob/main/examples/basic_usage.ipynb)
[![PyPI version](https://badge.fury.io/py/topic-wizard.svg)](https://pypi.org/project/topic-wizard/)
[![pip downloads](https://img.shields.io/pypi/dm/topic-wizard.svg)](https://pypi.org/project/topic-wizard/)
[![python version](https://img.shields.io/badge/Python-%3E=3.8-blue)](https://github.com/centre-for-humanities-computing/tweetopic)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
<br>


https://github.com/x-tabdeveloping/topicwizard/assets/13087737/9736f33c-6865-4ed4-bc17-d8e6369bda80



## New in version 0.5.0 🌟 

- Enhanced readibility and legibility of graphs.
- Added helper tooltips to help you understand and interpret the graphs.
- Improved stability.
- Negative topic distributions are now supported in documents.

## Features

-   Investigate complex relations between topics, words, documents and groups/genres/labels
-   Easy to use pipelines that can be utilized for downstream tasks
-   Sklearn, Gensim and BERTopic compatible :nut_and_bolt:
-   Highly interactive web app
-   Interactive and composable Plotly figures
-   Automatically infer topic names, oooor...
-   Name topics manually
-   Easy deployment :earth_africa:

## Installation

Install from PyPI:

```bash
pip install topic-wizard
```

## [Pipelines](https://x-tabdeveloping.github.io/topicwizard/usage.pipelines.html)

The main abstraction of topicwizard around a topic model is a topic pipeline, which consists of a vectorizer, that turns texts into bag-of-tokens
representations and a topic model which decomposes these representations into vectors of topic importance.
topicwizard allows you to use both scikit-learn pipelines or its own `TopicPipeline`.

<img align="right" width="300" src="https://x-tabdeveloping.github.io/topicwizard/_images/pipeline.png">


Let's build a pipeline. We will use scikit-learns CountVectorizer as our vectorizer component:
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=5, max_df=0.8, stop_words="english")
```
The topic model I will use for this example is Non-negative Matrix Factorization as it is fast and usually finds good topics.
```python
from sklearn.decomposition import NMF

model = NMF(n_components=10)
```
Then let's put this all together in a pipeline. You can either use sklearn Pipelines...
```python
from sklearn.pipeline import make_pipeline

topic_pipeline = make_pipeline(vectorizer, model)
```

Or topicwizard's [TopicPipeline](https://x-tabdeveloping.github.io/topicwizard/usage.pipelines.html#topicpipeline)

```python
from topicwizard.pipeline import make_topic_pipeline

topic_pipeline = make_topic_pipeline(vectorizer, model)
```

Let's load a corpus that we would like to analyze, in this example I will use 20newsgroups from sklearn.

```python
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset="all")
corpus = newsgroups.data

# Sklearn gives the labels back as integers, we have to map them back to
# the actual textual label.
group_labels = [newsgroups.target_names[label] for label in newsgroups.target]
```

Then let's fit our pipeline to this data:
```python
topic_pipeline.fit(corpus)
```

## [Web Application](https://x-tabdeveloping.github.io/topicwizard/application.html)

You can launch the topic wizard web application for interactively investigating your topic models. The app is also quite easy to [deploy](https://x-tabdeveloping.github.io/topicwizard/usage.deployment.html) in case you want to create a client-facing interface.

```python
import topicwizard

topicwizard.visualize(corpus, pipeline=topic_pipeline)
```

From version 0.3.0 you can also disable pages you do not wish to display thereby sparing a lot of time for yourself:

```python
# A large corpus takes a looong time to compute 2D projections for so
# so you can speed up preprocessing by disabling it alltogether.
topicwizard.visualize(corpus, pipeline=topic_pipeline, exclude_pages=["documents"])
```
| [Topics](https://x-tabdeveloping.github.io/topicwizard/usage.topics.html) | [Words](https://x-tabdeveloping.github.io/topicwizard/usage.words.html) | [Documents](https://x-tabdeveloping.github.io/topicwizard/usage.documents.html) | [Groups](https://x-tabdeveloping.github.io/topicwizard/usage.groups.html) |
| :----: | :----: | :----: | :----: |
| ![topics screenshot](assets/screenshot_topics.png) | ![words screenshot](assets/screenshot_words.png)  | ![documents screenshot](assets/screenshot_documents.png) | ![groups screenshot](docs/_static/screenshot_groups.png) |

## [Figures](https://x-tabdeveloping.github.io/topicwizard/api_reference.html#module-topicwizard.figures)

If you want customizable, faster, html-saveable interactive plots, you can use the figures API.
Here are a couple of examples:

```python
from topicwizard.figures import word_map, document_topic_timeline, topic_wordclouds, word_association_barchart
```

| Word Map | Timeline of Topics in a Document | 
| :----: | :----: |
| `word_map(corpus, pipeline=topic_pipeline)` | `document_topic_timeline( "Joe Biden takes over presidential office from Donald Trump.", pipeline=topic_pipeline)` |
| ![word map screenshot](assets/word_map.png) | ![doc_timeline](https://github.com/x-tabdeveloping/topic-wizard/assets/13087737/cf1faceb-e8ef-411f-80cd-a2a58befcf99) |

| Wordclouds of Topics | Topic for Word Importance |
| :----: | :----: |
| `topic_wordclouds(corpus, pipeline=topic_pipeline)` | `word_association_barchart(["supreme", "court"], corpus=corpus, pipeline=topic_pipeline)` |
| ![wordclouds](assets/topic_wordclouds.png) | ![topic_word_imp](https://github.com/x-tabdeveloping/topic-wizard/assets/13087737/0767b631-9e83-42cf-8796-8536abc486d0) |

For more information consult our [Documentation](https://x-tabdeveloping.github.io/topicwizard/index.html)
