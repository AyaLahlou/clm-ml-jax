"""Configuration loader for jax-agents."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Default config path (go up from utils -> jax_agents -> src -> project root)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config.yaml"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_llm_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get LLM configuration.
    
    Args:
        config: Full config dict. If None, loads from default path.
        
    Returns:
        LLM configuration dictionary
    """
    if config is None:
        config = load_config()
    
    return config.get("llm", {
        "model": "claude-sonnet-4-5",
        "temperature": 0.0,
        "max_tokens": 48000,
        "timeout": 600,
    })


def get_agent_config(agent_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific agent.
    
    Args:
        agent_name: Name of the agent (e.g., "static_analysis", "translator")
        config: Full config dict. If None, loads from default path.
        
    Returns:
        Agent configuration dictionary
    """
    if config is None:
        config = load_config()
    
    agents_config = config.get("agents", {})
    return agents_config.get(agent_name, {})

