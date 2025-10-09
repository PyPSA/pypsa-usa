"""
Path resolution utilities for PyPSA-USA workflow.

This module provides functions to resolve paths to bundled data files
and workflow components using importlib.resources.
"""

import importlib.resources as resources
from pathlib import Path
from typing import Union


def get_repo_data_path(relative_path: str = "") -> Path:
    """
    Get path to bundled repo_data files.
    
    Args:
        relative_path: Relative path within repo_data directory
        
    Returns:
        Path to the requested file or directory
        
    Example:
        >>> get_repo_data_path("costs/efs_tech_costs.csv")
        Path('/path/to/package/data/repo_data/costs/efs_tech_costs.csv')
    """
    package_path = resources.files('pypsa_usa.data.repo_data')
    if relative_path:
        return package_path / relative_path
    return package_path


def get_workflow_path() -> Path:
    """
    Get path to workflow directory containing Snakefile and rules.
    
    Returns:
        Path to the workflow directory
    """
    return resources.files('pypsa_usa.workflow')


def get_config_path(config_name: str) -> Path:
    """
    Get path to bundled config files.
    
    Args:
        config_name: Name of config file (e.g., 'config.default.yaml')
        
    Returns:
        Path to the config file
    """
    return resources.files('pypsa_usa.workflow.config') / config_name


def get_user_workspace_path(relative_path: str = "") -> Path:
    """
    Get path to user workspace directory.
    
    Args:
        relative_path: Relative path within user workspace
        
    Returns:
        Path to the requested location in user workspace
    """
    workspace = Path("user_workspace")
    if relative_path:
        return workspace / relative_path
    return workspace


def get_user_config_path(config_name: str) -> Path:
    """
    Get path to user config file.
    
    Args:
        config_name: Name of config file
        
    Returns:
        Path to user config file
    """
    return get_user_workspace_path("config") / config_name


def get_user_data_path(relative_path: str = "") -> Path:
    """
    Get path to user data directory.
    
    Args:
        relative_path: Relative path within user data directory
        
    Returns:
        Path to user data location
    """
    return get_user_workspace_path("data") / relative_path


def get_user_results_path(relative_path: str = "") -> Path:
    """
    Get path to user results directory.
    
    Args:
        relative_path: Relative path within user results directory
        
    Returns:
        Path to user results location
    """
    return get_user_workspace_path("results") / relative_path


def get_user_resources_path(relative_path: str = "") -> Path:
    """
    Get path to user resources directory.
    
    Args:
        relative_path: Relative path within user resources directory
        
    Returns:
        Path to user resources location
    """
    return get_user_workspace_path("resources") / relative_path


def get_user_logs_path(relative_path: str = "") -> Path:
    """
    Get path to user logs directory.
    
    Args:
        relative_path: Relative path within user logs directory
        
    Returns:
        Path to user logs location
    """
    return get_user_workspace_path("logs") / relative_path
