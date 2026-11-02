import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from preprocessing.split import split_ratings
from evaluation.metrics import ranking_metrics
from run_experiments import MODEL_NAMES, make_model
from config import Config
from evaluation.evaluator import evaluate
from algorithms.collaborative.user_knn import overlap_similarity


def fixture_ratings():
    return pd.DataFrame([(u, i, 1 + (u + i) % 5) for u in range(5) for i in range(12)],
                        columns=['user', 'item', 'rating'])


def test_split_disjoint_reproducible_and_complete():
    ratings = fixture_ratings()
    first = split_ratings(ratings)
    second = split_ratings(ratings)
    sets = [set(zip(part.user, part.item)) for part in first]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert sum(map(len, sets)) == len(ratings)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


def test_ranking_known_example():
    actual = ranking_metrics([5, 7, 9], {5, 9}, 3)
    assert actual['precision'] == pytest.approx(2 / 3)
    assert actual['recall'] == 1
    assert actual['ndcg'] == pytest.approx(1.5 / (1 + 1 / np.log2(3)))


@pytest.mark.parametrize('name', MODEL_NAMES)
def test_models_finite_and_bounded(name):
    ratings = fixture_ratings()
    features = csr_matrix(np.random.default_rng(2).uniform(size=(13, 6)))
    from sklearn.preprocessing import normalize
    features = normalize(features)
    model = make_model(name, Config(epochs=2)).fit(ratings, 5, 13, features)
    values = model.score(0, np.array([0, 6, 12]))
    assert values.shape == (3,)
    assert np.isfinite(values).all()
    if model.rating_model:
        assert ((values >= 1) & (values <= 5)).all()
    elif name != 'popularity':
        assert ((values >= 0) & (values <= 1)).all()


def test_single_class_content_fallback():
    ratings = fixture_ratings()
    ratings['rating'] = 5.0
    for name in ('decision_tree', 'naive_bayes'):
        model = make_model(name, Config()).fit(ratings, 5, 12, csr_matrix(np.eye(12)))
        assert np.isfinite(model.score(0, np.array([1, 2]))).all()


def test_factor_model_learns_observed_signal():
    ratings = fixture_ratings()
    model = make_model('matrix_factorization', Config(epochs=60)).fit(ratings, 5, 12)
    error = np.mean((model.predict(ratings.user, ratings.item) - ratings.rating) ** 2)
    baseline = np.mean((ratings.rating - ratings.rating.mean()) ** 2)
    assert error < baseline


def test_recommendations_exclude_observed_items():
    train, validation, test = split_ratings(fixture_ratings())
    final_train = pd.concat([train, validation], ignore_index=True)
    features = csr_matrix(np.eye(12))
    model = make_model('popularity', Config()).fit(final_train, 5, 12, features)
    users = test.loc[test.rating >= 4, 'user'].unique()
    metrics, examples = evaluate(model, final_train, test, users, features, Config(k=2))
    for example in examples:
        seen = set(final_train.loc[final_train.user == example['user'], 'item'])
        assert not seen.intersection(example['items'])
    assert metrics['evaluated_users'] == len(users)


def test_pearson_uses_overlap_and_rejects_insufficient_support():
    ratings = csr_matrix(np.array([[1., 2., 3., 0.], [2., 3., 4., 5.], [5., 0., 0., 0.]]))
    similarity = overlap_similarity(ratings, shrinkage=0)
    assert similarity[0, 1] == pytest.approx(1)
    assert similarity[0, 2] == 0
    assert np.diag(similarity).sum() == 0

