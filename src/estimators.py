"""Small sklearn-compatible estimator adapters used by the model suite."""

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Allow numeric-target classifiers to preserve the project's string labels."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        self.estimator_ = clone(self.estimator)
        if self.estimator_.__class__.__module__.startswith("xgboost"):
            multiclass = len(self.classes_) > 2
            self.estimator_.set_params(
                objective="multi:softprob" if multiclass else "binary:logistic",
                eval_metric="mlogloss" if multiclass else "logloss",
            )
        self.estimator_.fit(X, self.label_encoder_.transform(y))
        return self

    def predict(self, X):
        return self.label_encoder_.inverse_transform(self.estimator_.predict(X).astype(int))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)
