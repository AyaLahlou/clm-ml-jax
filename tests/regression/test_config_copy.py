"""Each run writes the four files of its experiment, as the model read them, into
``OUTPUT/config``: a folder that repeats the run."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression


def test_output_holds_a_copy_of_the_experiment(current):
    from clm_ml.experiment import load_experiment

    run, _, out = current
    copy = load_experiment(out / "config")
    assert (copy.domain_file, copy.config) == (run.domain_file, run.config)
    assert (copy.start, copy.n_steps) == (run.start, run.n_steps)
    # every path absolute, pointing at the same files
    met = copy.run_file.forcing.meteorology.files[0]
    assert met.startswith("/") and copy.path(met) == run.path(
        run.run_file.forcing.meteorology.files[0]
    )
    assert copy.output_directory == run.output_directory
    for k in run.params.values:
        np.testing.assert_array_equal(copy.params.values[k], run.params.values[k])
