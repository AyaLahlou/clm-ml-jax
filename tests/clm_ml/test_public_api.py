"""The public API: the names ``import clm_ml`` offers."""

import clm_ml

PUBLIC = ["Experiment", "Model", "load_experiment", "new_experiment", "upgrade_experiment"]


def test_public_names():
    assert sorted(clm_ml._EXPORTS) == PUBLIC
    for name in PUBLIC:
        assert getattr(clm_ml, name) is not None
