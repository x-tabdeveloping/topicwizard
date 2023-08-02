from typing import Iterable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.preprocessing import normalize

from topicwizard.prepare.topics import infer_topic_names


class TopicPipeline(Pipeline):
    def __init__(
        self,
        steps: list[tuple[str, BaseEstimator]],
        *,
        memory=None,
        verbose=False,
        pandas_out=False,
        norm_row=True,
    ):
        super().__init__(steps, memory=memory, verbose=verbose)
        self.topic_names = None
        self.pandas_out = pandas_out
        self.norm_row = norm_row
        if len(self) < 2:
            raise ValueError(
                "A Topic pipeline should at least have a vectorizer and a topic model."
            )
        _, self.vectorizer_ = self.steps[0]
        _, self.topic_model_ = self.steps[-1]
        if not hasattr(self.topic_model_, "transform"):
            raise TypeError("A topic model should have a transform method.")
        if not hasattr(self.vectorizer_, "transform"):
            raise TypeError("A vectorizer should have a transform method.")

    def _validate(self):
        if not hasattr(self.vectorizer_, "get_feature_names_out"):
            raise TypeError(
                "A fitted vectorizer should have a get_feature_names_out method."
            )
        if not hasattr(self.topic_model_, "components_"):
            raise TypeError("A fitted topic model should have a components_ attribute.")

    def fit(self, X: Iterable[str], y=None):
        super().fit(X, y)
        self._validate()
        self.topic_names = infer_topic_names(self)
        return self

    def partial_fit(self, X, y=None, classes=None, **kwargs):
        """
        Fits the components, but allow for batches.
        """
        for name, step in self.steps:
            if not hasattr(step, "partial_fit"):
                raise ValueError(
                    f"Step {name} is a {step} which does not have `.partial_fit` implemented."
                )
        for name, step in self.steps:
            if hasattr(step, "predict"):
                step.partial_fit(X, y, classes=classes, **kwargs)
            else:
                step.partial_fit(X, y)
            if hasattr(step, "transform"):
                X = step.transform(X)
        self._validate()
        self.topic_names = infer_topic_names(self)
        return self

    def transform(self, X: Iterable[str]):
        if self.topic_names is None:
            raise NotFittedError("Topic pipeline has not been fitted yet.")
        X_new = super().transform(X)
        if self.norm_row:
            X_new = normalize(X_new, norm="l1", axis=1)
        if self.pandas_out:
            return pd.DataFrame(X_new, columns=self.topic_names)
        else:
            return X_new

    def get_feature_names_out(self):
        return self.topic_names

    def fit_transform(self, X: Iterable[str], y=None):
        return self.fit(X, y).transform(X)

    def set_output(self, transform=None):
        if transform == "pandas":
            self.pandas_out = True
        return self


def make_topic_pipeline(
    *steps, memory=None, verbose=False, pandas_out=True, norm_row=True
):
    return TopicPipeline(
        _name_estimators(steps),
        memory=memory,
        verbose=verbose,
        pandas_out=pandas_out,
        norm_row=norm_row,
    )
