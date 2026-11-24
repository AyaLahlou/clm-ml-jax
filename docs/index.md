# CLM-ML-JAX

CLM-ml v2 is the multilayer canopy model developed for the Community Land Model by Gordon
Bonan (Bonan et al. 2021, 2026). CLM-ML-JAX is a JAX version of it: it gives the same results
as the Fortran model ([gbonan/CLM-ml_v2.CHATS](https://github.com/gbonan/CLM-ml_v2.CHATS)) and
adds

- **gradients** of any output with respect to parameters, forcing and initial state, for
  sensitivity analysis and calibration;
- **GPU execution**, and runs from one flux tower up to the whole globe;
- **configuration in four readable files** per experiment, with every setting explained.

The canopy is resolved in layers from the ground to the forcing height, with sunlit and shaded
leaves in each, a roughness-sublayer turbulence closure, leaf photosynthesis and stomatal
optimization coupled to plant hydraulics, and an implicit solution of the air temperature,
humidity and leaf temperature profiles.

**Start here:** [Getting started](getting_started.md) runs one day at the CHATS walnut orchard
in a few minutes. Then [your own tower](your_own_tower.md), the
[examples](examples.md) (including gradients), and the [settings reference](reference/settings.md).

```{toctree}
:maxdepth: 2
:hidden:

getting_started
your_own_tower
examples
how_the_model_works
reference/settings
```

## Citation

Please cite the software ([`CITATION.cff`](repo:CITATION.cff)) and the model it translates:

- Bonan, G. B., Patton, E. G., Finnigan, J. J., Baldocchi, D. D., & Harman, I. N. (2021).
  Moving beyond the incorrect but useful paradigm: reevaluating big-leaf and multilayer plant
  canopies to model biosphere-atmosphere fluxes – a review. *Agricultural and Forest
  Meteorology*, 306, 108435. <https://doi.org/10.1016/j.agrformet.2021.108435>
- Bonan, G. B., Burns, S. P., & Patton, E. G. (2026). Beyond surface fluxes: Observational and
  computational needs of multilayer canopy models – A walnut orchard test case. *Agricultural
  and Forest Meteorology*, 378, 110960. <https://doi.org/10.1016/j.agrformet.2025.110960>

Source code, issues and the development record:
[github.com/linniahawkins/clm-ml-jax](https://github.com/linniahawkins/clm-ml-jax).
