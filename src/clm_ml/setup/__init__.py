"""The model's setup as pure functions: every patch's constants
(:class:`clm_ml.step.PatchConstants`) and initial state (:class:`clm_ml.state.State`) from
the domain, the parameters, the model options and the first step's forcing, without
module globals.

- :mod:`clm_ml.setup.soil`: soil layers, bedrock, soil properties, roots, soil albedos.
- :mod:`clm_ml.setup.canopy`: leaf/stem area profiles, layer grid, initial canopy state.
- :mod:`clm_ml.setup.patches`: the patches' constants and initial state.
- :mod:`clm_ml.setup.tables`: data tables.
"""
