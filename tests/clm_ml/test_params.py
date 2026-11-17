"""params.yaml loading (the built-in files equalled the legacy PFT tables exactly when the
legacy tables were deleted, Phase 7.4)."""

import subprocess
import sys

import numpy as np
import pytest

from clm_ml.io.yaml_schema import ConfigError
from clm_ml.params import PFT_NAMES, Params

from .conftest import SRC_DIR

WALNUT = "broadleaf_deciduous_temperate_tree"


def test_chats_values_match_paper_table_2():
    """Bonan, Burns & Patton (2026), Table 2, CHATS column."""
    p = Params.from_yaml("chats")
    expected = {
        "vcmax25": 125.0, "iota": 375.0, "psi50_gs": -1.6, "shape_gs": 40.0,
        "plant_conductance": 7.0, "root_resistivity": 14.0, "leaf_angle_index": 0.53,
    }  # fmt: skip
    for name, value in expected.items():
        assert p.get(WALNUT, name) == value, name
    np.testing.assert_array_equal(p.get(WALNUT, "leaf_reflectance"), [0.06, 0.42])
    np.testing.assert_array_equal(p.get(WALNUT, "leaf_transmittance"), [0.04, 0.43])
    np.testing.assert_array_equal(p.get(WALNUT, "stem_reflectance"), [0.21, 0.49])


@pytest.mark.parametrize("name", ["chats", "clm5_defaults", "clm6"])
def test_stable_zeta_limit_of_the_compsets(name):
    """Decision 123: CLM's 0.5 for shrubs, grasses and crops, no limit for trees."""
    from clm_ml.params import NO_ZETA_LIMIT, ZETA_MAX_STABLE_DEFAULT
    from clm_ml.processes import turbulence

    assert NO_ZETA_LIMIT == turbulence.NO_ZETA_LIMIT
    p = Params.from_yaml(name)
    for pft in PFT_NAMES[1:]:
        expected = NO_ZETA_LIMIT if pft.endswith("_tree") else 0.5
        assert p.get(pft, "zeta_max_stable") == expected == ZETA_MAX_STABLE_DEFAULT[pft]


def test_a_copy_edited_is_a_new_parameter_set(tmp_path):
    """The way to make a parameter set: a complete copy (`clm-ml params chats -o`), edited."""
    import yaml

    from clm_ml.params_convert import write_params_yaml

    path = write_params_yaml("chats", tmp_path / "mine.yaml")
    data = yaml.safe_load(path.read_text())
    data["pft"][WALNUT]["vcmax25"] = 100.0
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    p = Params.from_yaml(path)
    assert p.get(WALNUT, "vcmax25") == 100.0
    assert p.get(WALNUT, "iota") == 375.0
    assert p.get("c4_grass", "g0_ball_berry") == 0.04
    assert np.isnan(p.values["vcmax25"][PFT_NAMES.index("not_vegetated")])


def test_params_files_are_complete(tmp_path):
    """No inheritance and no filled-in defaults: `extends` and missing values are errors."""
    (tmp_path / "ext.yaml").write_text(f"extends: chats\npft:\n  {WALNUT}:\n    vcmax25: 1.0\n")
    with pytest.raises(ConfigError, match="'extends' is not supported.*clm-ml params chats"):
        Params.from_yaml(tmp_path / "ext.yaml")
    (tmp_path / "part.yaml").write_text(f"pft:\n  {WALNUT}:\n    vcmax25: 1.0\n")
    with pytest.raises(ConfigError, match="every parameter of every PFT 1-16.*missing"):
        Params.from_yaml(tmp_path / "part.yaml")


@pytest.mark.parametrize(
    "body, message",
    [
        (f"pft:\n  {WALNUT}:\n    vcmax: 1.0\n", "unknown parameter"),
        ("pft:\n  walnut:\n    vcmax25: 1.0\n", "unknown PFT 'walnut'"),
        (f"pft:\n  {WALNUT}:\n    leaf_reflectance: 0.1\n", "expected [vis, nir]"),
        ("extends: mine.yaml\n", "'extends' is not supported"),
        ("pfts: {}\n", "unknown key(s) ['pfts']"),
    ],
)
def test_invalid_params(tmp_path, body, message):
    (tmp_path / "mine.yaml").write_text(body)
    with pytest.raises(ConfigError) as exc:
        Params.from_yaml(tmp_path / "mine.yaml")
    assert message in str(exc.value)


def test_set_returns_copy():
    p = Params.from_yaml("chats")
    q = p.set(WALNUT, vcmax25=90.0)
    assert (p.get(WALNUT, "vcmax25"), q.get(WALNUT, "vcmax25")) == (125.0, 90.0)


# --- clm-ml params: CLM parameter files and flattened params files (Phase 6b) -------------


def _fake_clm_parameter_file(path, n_pft=20):
    """A small CLM-style parameter file: every converted variable, value = 0.01 * row + k."""
    import netCDF4 as nc

    from clm_ml.params_convert import CLM_PARAMETER_FILE_FIELDS

    with nc.Dataset(path, "w") as ds:
        ds.createDimension("pft", n_pft)
        ds.createDimension("string_length", 40)
        ds.createDimension("variants", 2)
        names = list(PFT_NAMES) + [f"crop_{i}" for i in range(n_pft - len(PFT_NAMES))]
        v = ds.createVariable("pftname", "S1", ("pft", "string_length"))
        v[:] = np.array([[c.encode() for c in n.ljust(40)] for n in names], dtype="S1")
        variables = sorted({x for vs, _ in CLM_PARAMETER_FILE_FIELDS.values() for x in vs})
        for k, name in enumerate(variables):
            row = 0.01 * np.arange(n_pft) + k
            if name == "rootprof_beta":
                ds.createVariable(name, "f8", ("variants", "pft"))[:] = np.stack([row, row + 0.5])
            elif name == "c3psn":
                ds.createVariable(name, "f8", ("pft",))[:] = (np.arange(n_pft) != 14).astype(float)
            else:
                ds.createVariable(name, "f8", ("pft",))[:] = row
    return variables


def test_params_from_a_clm_parameter_file(tmp_path):
    """The converted table: CLM-file values for what CLM defines (units converted, the water
    root-profile variant), the base's for the CLM-ml-specific parameters; complete."""
    from clm_ml.params import PARAM_SPECS
    from clm_ml.params_convert import write_params_yaml

    variables = _fake_clm_parameter_file(tmp_path / "clm.nc")
    out = write_params_yaml(str(tmp_path / "clm.nc"), tmp_path / "p.yaml")
    p, base = Params.from_yaml(out), Params.from_yaml("clm5_defaults")
    k = {name: i for i, name in enumerate(variables)}
    row = PFT_NAMES.index(WALNUT)
    assert p.get(WALNUT, "leaf_dimension") == 0.01 * row + k["dleaf"]
    assert p.get(WALNUT, "g0_medlyn") == (0.01 * row + k["medlynintercept"]) * 1e-6
    assert p.get(WALNUT, "root_profile_beta") == 0.01 * row + k["rootprof_beta"]  # variant 1
    np.testing.assert_array_equal(
        p.get(WALNUT, "leaf_reflectance"), [0.01 * row + k["rholvis"], 0.01 * row + k["rholnir"]]
    )
    assert p.get(WALNUT, "z0m_to_height_ratio") == 0.01 * row + k["z0mr"]
    assert p.get("c4_grass", "c3_photosynthesis") == 0.0
    for name in ("vcmax25", "iota", "plant_conductance", "gs_min"):  # CLM-ml specific
        assert p.get(WALNUT, name) == base.get(WALNUT, name)
    every = [s.name for s in PARAM_SPECS]
    assert not p.missing_for(list(range(1, len(PFT_NAMES))), every)
    text = out.read_text()
    assert "extends" not in text and "clm.nc" in text  # standalone, source recorded


def test_copy_of_a_params_file_is_identical(tmp_path):
    from clm_ml.params_convert import write_params_yaml

    out = write_params_yaml("chats", tmp_path / "chats_copy.yaml")
    a, b = Params.from_yaml("chats"), Params.from_yaml(out)
    for name in a.values:
        np.testing.assert_array_equal(a.values[name], b.values[name])


def test_clm_parameter_files_are_not_read_by_params_yaml(tmp_path):
    (tmp_path / "p.yaml").write_text("clm_parameters: x.nc\n")
    with pytest.raises(ConfigError, match="clm-ml params"):
        Params.from_yaml(tmp_path / "p.yaml")


def test_builtin_clm6_is_complete():
    """data/params/clm6.yaml (the CTSM 6.0 file converted): every parameter of every PFT,
    including the roughness ratios grid runs need."""
    from clm_ml.params import PARAM_SPECS

    p = Params.from_yaml("clm6")
    assert not p.missing_for(list(range(1, len(PFT_NAMES))), [s.name for s in PARAM_SPECS])
    assert p.get("needleleaf_evergreen_temperate_tree", "z0m_to_height_ratio") == 0.055
    assert p.get("needleleaf_evergreen_temperate_tree", "leaf_dimension") == 0.000648
