"""Feature blocks carried forward from the original GDP_PREDICTION notebook.

The blocks bridge the original ML project and the refactored econometric workflow.
They do not imply that every member belongs in one regression: mechanically related
measures are alternative representations of the same economic concept.
"""
from __future__ import annotations

from collections.abc import Mapping

FEATURE_BLOCKS: Mapping[str, tuple[str, ...]] = {
    "capital": ("ppp_capital", "priv_capital", "gov_capital"),
    "labor": (
        "labor_force", "human_capital_index", "unemployment_pct",
        "effective_labor", "labor_force_population_ratio",
        "employment_adjusted_effective_labor",
    ),
    "technology": (
        "tfp_index", "researchers_per_million", "total_patents",
        "energy_per_labor",
    ),
    "demographics": (
        "population_total", "age_dependency_ratio", "urban_population_pct",
    ),
}

ORIGINAL_MODEL_SETS: Mapping[str, tuple[str, ...]] = {
    "set1_raw_blocks": (
        "ppp_capital", "priv_capital", "gov_capital",
        "labor_force", "human_capital_index",
        "tfp_index", "researchers_per_million", "energy_per_labor",
    ),
    "set2_technology_demographic_pca": (
        "priv_capital", "gov_capital", "ppp_capital", "labor_force",
        "technology_PC1", "demographics_PC1",
    ),
    "set3_labor_technology_demographic_pca": (
        "priv_capital", "gov_capital", "ppp_capital",
        "labor_PC1", "technology_PC1", "demographics_PC1",
    ),
    "set4_all_block_pca": (
        "capital_PC1", "labor_PC1", "technology_PC1", "demographics_PC1",
    ),
}

def block_members(block: str) -> tuple[str, ...]:
    """Return the documented members of one original-notebook feature block."""
    return FEATURE_BLOCKS[block]
