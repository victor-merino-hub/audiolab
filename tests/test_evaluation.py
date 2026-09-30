import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from audiolab.evaluation import evaluate_on_folds, random_folds


def test_random_folds_are_balanced():
    y = np.repeat(np.arange(4), 10)              # 4 classes x 10 samples
    folds = random_folds(y, n_folds=5, seed=0)
    assert set(folds) == {1, 2, 3, 4, 5}
    for k in range(1, 6):
        assert np.bincount(y[folds == k]).tolist() == [2, 2, 2, 2]


def test_evaluate_on_folds_separable_data():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 20)
    X = y[:, None] * 10 + rng.normal(size=(40, 2))     # two far-apart clusters
    scores = evaluate_on_folds(KNeighborsClassifier(1), X, y, random_folds(y))
    assert scores.shape == (5,)
    assert np.all(scores == 1.0)


def test_evaluate_on_folds_never_tests_on_training_data():
    # Every sample is unique noise with a random label: memorizing it is useless
    # on unseen samples, so accuracy must stay near chance (0.5), not 1.0
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = rng.integers(0, 2, size=200)
    scores = evaluate_on_folds(KNeighborsClassifier(1), X, y, random_folds(y))
    assert 0.3 < scores.mean() < 0.7
