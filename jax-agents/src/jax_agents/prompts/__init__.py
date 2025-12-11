"""Prompt templates for JAX-CTSM translation agents."""

from jax_agents.prompts.analysis_prompts import ANALYSIS_PROMPTS
from jax_agents.prompts.translation_prompts import TRANSLATION_PROMPTS
from jax_agents.prompts.orchestrator_prompts import ORCHESTRATOR_PROMPTS
from jax_agents.prompts.test_prompts import TEST_PROMPTS

__all__ = [
    "ANALYSIS_PROMPTS",
    "TRANSLATION_PROMPTS",
    "ORCHESTRATOR_PROMPTS",
    "TEST_PROMPTS",
]

