"""domain.yaml towers and the Domain (gridcells, patches, soil columns)."""

import numpy as np
import pytest
import yaml

from clm_ml.config import ModelConfig
from clm_ml.domain import Domain, load_towers
from clm_ml.experiment import compsets_dir
from clm_ml.io.yaml_schema import ConfigError
from clm_ml.params import Params

PARAMS = Params.from_yaml("chats")
CHATS7 = load_towers(compsets_dir() / "towers" / "chats7.yaml")


def test_tower_compsets_hold_chats7_only():
    assert sorted(p.name for p in (compsets_dir() / "towers").glob("*.yaml")) == ["chats7.yaml"]
    assert list(CHATS7) == ["CHATS7"]


def test_chats7_domain_and_layer_count():
    domain = Domain.from_towers(CHATS7, params=PARAMS, config=ModelConfig())
    assert (domain.n_gridcells, domain.n_patches) == (1, 1)
    assert domain.patches.weight.tolist() == [1.0]
    # 20 layers within the 10 m canopy at 0.5 m, 26 above it to the 23 m measurement height.
    assert domain.layer_counts(ModelConfig()) == [46]


def test_too_few_max_layers_names_the_patch():
    config = ModelConfig.from_dict({"canopy_layers": {"max_layers": 40}})
    with pytest.raises(ConfigError, match="CHATS7 patch 0 .* needs 46 canopy layers"):
        Domain.from_towers(CHATS7, params=PARAMS, config=config)


def test_explicit_layer_counts():
    config = ModelConfig.from_dict({"canopy_layers": {"layer_counts": {"within": 10, "above": 5}}})
    domain = Domain.from_towers(CHATS7, params=PARAMS, config=config)
    assert domain.layer_counts(config) == [15]


MIXED = """
MIXED:
  latitude: 45.0
  longitude: -85.0
  measurement_height: 30.0
  soil: {texture: {given_as: class, texture_class: loam}, organic_matter: 20.0}
  patches:
    - {pft: broadleaf_deciduous_temperate_tree, weight: 0.7, canopy_height: 21.0,
       root_biomass: 500.0}
    - {pft: c3_non-arctic_grass, weight: 0.3, canopy_height: 0.5, root_biomass: 400.0}
"""


def _towers(tmp_path, text=MIXED):
    path = tmp_path / "towers.yaml"
    path.write_text(text)
    return load_towers(path)


# With the default spacing a 0.5 m grass patch gets 0.1 m layers all the way up to the 30 m
# measurement height (300 layers), so mixed towers set layer counts explicitly.
EXPLICIT_LAYERS = ModelConfig.from_dict(
    {"canopy_layers": {"layer_counts": {"within": 10, "above": 10}}}
)


def test_short_canopy_with_default_spacing_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="c3_non-arctic_grass\\) needs 300 canopy layers"):
        Domain.from_towers(
            _towers(tmp_path),
            params=PARAMS,
            config=ModelConfig.from_dict({"canopy_layers": {"max_layers": 60}}),  # tree needs 60
        )


def test_tower_with_two_patches(tmp_path):
    domain = Domain.from_towers(_towers(tmp_path), params=PARAMS, config=EXPLICIT_LAYERS)
    assert domain.n_patches == 2
    np.testing.assert_array_equal(domain.patches.gridcell_index, [0, 0])
    np.testing.assert_array_equal(domain.patches.column_index, [0, 1])
    np.testing.assert_allclose(domain.patches.weight, [0.7, 0.3])


def test_min_patch_weight_drops_and_renormalizes(tmp_path):
    domain = Domain.from_towers(_towers(tmp_path), params=PARAMS, config=EXPLICIT_LAYERS,
                                min_patch_weight=0.5)  # fmt: skip
    assert domain.n_patches == 1
    assert domain.patches.weight.tolist() == [1.0]


def test_towers_in_order_of_the_file(tmp_path):
    towers = dict(_towers(tmp_path), **CHATS7)
    domain = Domain.from_towers(towers, params=PARAMS, config=EXPLICIT_LAYERS)
    assert domain.gridcells.names == ("MIXED", "CHATS7")


@pytest.mark.parametrize(
    "text, message",
    [
        (MIXED.replace("0.3", "0.4"), "weights sum to 1.1"),
        (MIXED.replace("c3_non-arctic_grass", "grass"), "unknown PFT 'grass'"),
        (MIXED.replace("  latitude: 45.0\n", ""), "MIXED.latitude: required key is missing"),
        (MIXED.replace(", root_biomass: 400.0", ""), "root_biomass"),
        (
            MIXED.replace("given_as: class, texture_class: loam", "given_as: class"),
            "texture.texture_class",
        ),
        (MIXED.replace("texture_class: loam", "texture_class: peat"), "peat"),
    ],
)
def test_invalid_towers(tmp_path, text, message):
    with pytest.raises(ConfigError) as exc:
        _towers(tmp_path, text)
    assert message in str(exc.value)


def test_no_towers():
    with pytest.raises(ConfigError, match="list at least one tower"):
        Domain.from_towers({}, params=PARAMS, config=ModelConfig())


def test_chats7_compset_is_the_shipped_tower():
    data = yaml.safe_load((compsets_dir() / "towers" / "chats7.yaml").read_text())
    assert data["CHATS7"]["measurement_height"] == 23.0
