"""
JAX-CTSM Translation Agents.

Multi-agent system for converting Fortran CTSM code to JAX using Claude 4.5 Sonnet.
"""

from jax_agents.base_agent import BaseAgent
from jax_agents.translator import TranslatorAgent, TranslationResult
from jax_agents.test_agent import TestAgent, TestGenerationResult
from jax_agents.repair_agent import RepairAgent, RepairResult

__version__ = "0.1.0"

__all__ = [
    "BaseAgent",
    "TranslatorAgent",
    "TranslationResult",
    "TestAgent",
    "TestGenerationResult",
    "RepairAgent",
    "RepairResult",
]

