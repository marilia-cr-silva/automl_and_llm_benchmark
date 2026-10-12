"""Random-forest hyper-parameter search shared by the 04 stage scripts."""

from __future__ import annotations

from typing import Any


def search_space() -> dict[str, list]:
    """Return the RandomizedSearchCV search space of the 04 stages.

    Note: ``max_samples`` offers the strings ``"int"``/``"float"``. scikit-learn rejects them,
    so candidates that draw them fail to fit and are scored as failed.
    """
    return {
        "n_estimators": [10, 50, 100],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [5, 10, 15, 20, 30, 35, 40, 50, 100],
        "max_features": ["sqrt", "log2", None],
        "oob_score": [True, False],
        "random_state": [0, 5, 10, 20, 30, 42, 100],
        "max_samples": ["int", "float", None],
    }


def build_search(random_state: int = 42, scoring: str = "f1_micro", cv: int = 5) -> Any:
    """Return the (unfitted) ``RandomizedSearchCV`` over a ``RandomForestClassifier``."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    estimator = RandomForestClassifier(
        n_jobs=-1,
        verbose=0,
        class_weight=None,
        warm_start=False,
        ccp_alpha=0.0,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.05,
        bootstrap=True,
    )
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=search_space(),
        random_state=random_state,
        scoring=scoring,
        cv=cv,
    )
