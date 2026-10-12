import pytest

from automl_and_llm_benchmark.common import random_forest

EXPECTED_SEARCH_SPACE = {
    "n_estimators": [10, 50, 100],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [5, 10, 15, 20, 30, 35, 40, 50, 100],
    "max_features": ["sqrt", "log2", None],
    "oob_score": [True, False],
    "random_state": [0, 5, 10, 20, 30, 42, 100],
    "max_samples": ["int", "float", None],
}


def test_search_space_values():
    assert random_forest.search_space() == EXPECTED_SEARCH_SPACE


def test_search_space_returns_fresh_dict():
    random_forest.search_space()["n_estimators"].append(1)
    assert random_forest.search_space() == EXPECTED_SEARCH_SPACE


def test_build_search():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    search = random_forest.build_search()
    assert isinstance(search, RandomizedSearchCV)
    assert search.cv == 5
    assert search.random_state == 42
    assert search.scoring == "f1_micro"
    assert search.param_distributions == EXPECTED_SEARCH_SPACE
    est = search.estimator
    assert isinstance(est, RandomForestClassifier)
    params = est.get_params()
    for key, value in {
        "n_jobs": -1,
        "verbose": 0,
        "class_weight": None,
        "warm_start": False,
        "ccp_alpha": 0.0,
        "min_samples_leaf": 3,
        "min_weight_fraction_leaf": 0.0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.05,
        "bootstrap": True,
    }.items():
        assert params[key] == value


def test_build_search_overrides():
    pytest.importorskip("sklearn")
    search = random_forest.build_search(random_state=1, scoring="accuracy", cv=3)
    assert (search.random_state, search.scoring, search.cv) == (1, "accuracy", 3)
