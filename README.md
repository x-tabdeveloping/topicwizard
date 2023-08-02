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



https://user-images.githubusercontent.com/13087737/234209888-0d20ede9-2ea1-4d6e-b69b-71b863287cc9.mp4

## New in version 0.4.0 🌟 🌟

- Introduced topic pipelines that make it easier and safer to use topic models in downstream tasks and interpretation.

## New in version 0.3.1 🌟 🌟

- You can now investigate relations of pre-existing labels to your topics and words :mag:

## New in version 0.3.0 🌟 

 - Exclude pages, that are not needed :bird:
 - Self-contained interactive figures :gift:
 - Topic name inference is now default behavior and is done implicitly.


## Features

-   Investigate complex relations between topics, words, documents and groups/genres/labels
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

## Usage ([documentation](https://x-tabdeveloping.github.io/topic-wizard/))

### Step 0:

Have a corpus ready for analysis, in this example I am going to use 20 newgroups from scikit-learn.

```python
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset="all")
corpus = newsgroups.data

# Sklearn gives the labels back as integers, we have to map them back to
# the actual textual label.
group_labels = [newsgroups.target_names[label] for label in newsgroups.target]
```

### Step 1:

Train a scikit-learn compatible topic model.
(If you want to use non-scikit-learn topic models, check [compatibility](https://x-tabdeveloping.github.io/topic-wizard/usage.compatibility.html))

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# Create topic pipeline
pipeline = make_pipeline(
    CountVectorizer(stop_words="english", min_df=10),
    NMF(n_components=30),
)

# Then fit it on the given texts
pipeline.fit(corpus)
```

From version 0.4.0 you can also use TopicPipelines, which are almost functionally identical but come with a set of built-in conveniences and
safeties.

```python
from topicwizard.pipeline import make_topic_pipeline

pipeline = make_topic_pipeline(
    CountVectorizer(stop_words="english", min_df=10),
    NMF(n_components=30),
)
```

### Step 2a:

Visualize with the topicwizard webapp :bulb:

```python
import topicwizard

topicwizard.visualize(corpus, pipeline=pipeline)
```

From version 0.3.0 you can also disable pages you do not wish to display thereby sparing a lot of time for yourself:

```python
# A large corpus takes a looong time to compute 2D projections for so
# so you can speed up preprocessing by disabling it alltogether.
topicwizard.visualize(corpus, pipeline=pipeline, exclude_pages=["documents"])
```


![topics screenshot](assets/screenshot_topics.png)
![words screenshot](assets/screenshot_words.png)
![words screenshot](assets/screenshot_words_zoomed.png)
![documents screenshot](assets/screenshot_documents.png)

From version 0.3.1 you can investigate groups/labels by passing them along to the webapp.

```python
topicwizard.visualize(corpus, pipeline=pipeline, group_labels=group_labels)
```

![groups screenshot](docs/_static/screenshot_groups.png)

Ooooor...

### Step 2b:

Produce high quality self-contained HTML plots and create your own dashboards/reports :strawberry:

### Map of words

```python
from topicwizard.figures import word_map

word_map(corpus, pipeline=pipeline)
```

![word map screenshot](assets/word_map.png)

### Timelines of topic distributions

```python
from topicwizard.figures import document_topic_timeline

document_topic_timeline(
    "Joe Biden takes over presidential office from Donald Trump.",
    pipeline=pipeline,
)
```
![document timeline](assets/document_topic_timeline.png)

### Wordclouds of your topics :cloud:

```python
from topicwizard.figures import topic_wordclouds

topic_wordclouds(corpus, pipeline=pipeline)
```

![wordclouds](assets/topic_wordclouds.png)

#### And much more... ([documentation](https://x-tabdeveloping.github.io/topic-wizard/))
