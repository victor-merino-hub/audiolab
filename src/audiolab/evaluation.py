"""Model evaluation with cross-validation."""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


def predict_on_folds(model, X, y, folds):
    """Out-of-fold predictions: each sample is predicted by a model that never saw its fold."""
    pred = np.empty_like(y)
    for k in np.unique(folds):
        test = folds == k
        fitted = clone(model).fit(X[~test], y[~test])     # clone: a fresh, untrained copy
        pred[test] = fitted.predict(X[test])
    return pred


def fold_accuracies(y, pred, folds):
    """Accuracy of the predictions within each fold."""
    return np.array([np.mean(pred[folds == k] == y[folds == k]) for k in np.unique(folds)])


def evaluate_on_folds(model, X, y, folds):
    """For each fold: train on the other folds and test on it. Returns the accuracy per fold."""
    return fold_accuracies(y, predict_on_folds(model, X, y, folds), folds)


def random_folds(y, n_folds=5, seed=0):
    """Assign each sample to a random fold (balanced by class), ignoring where it came from."""
    folds = np.zeros(len(y), dtype=int)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        folds[test] = k
    return folds
