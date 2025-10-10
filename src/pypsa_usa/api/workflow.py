"""
Python API for PyPSA-USA workflow execution.

This module provides a high-level interface for running PyPSA-USA workflows
programmatically using the Snakemake Python API.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import snakemake

from ..workflow._paths import get_workflow_path


def get_user_config_path() -> Path:
    """
    Get the path to the user's PyPSA-USA configuration file.

    Returns
    -------
        Path to the user config file (typically ~/.config/pypsa-usa/config.json)
    """
    # Use XDG Base Directory Specification
    config_dir = Path.home() / ".config" / "pypsa-usa"
    return config_dir / "config.json"


def get_default_workspace() -> Path | None:
    """
    Get the user's default workspace from their configuration.

    Returns
    -------
        Path to default workspace if set, None otherwise
    """
    config_path = get_user_config_path()
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
        workspace_path = config.get("default_workspace")
        if workspace_path:
            return Path(workspace_path).expanduser().resolve()
    except (json.JSONDecodeError, KeyError, OSError):
        pass

    return None


def set_default_workspace(workspace: str | Path) -> Path:
    """
    Set the user's default workspace in their configuration.

    Args:
        workspace: Path to set as default workspace

    Returns
    -------
        Path to the resolved workspace
    """
    workspace_path = Path(workspace).expanduser().resolve()
    config_path = get_user_config_path()

    # Create config directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new one
    config = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            config = {}

    # Update default workspace
    config["default_workspace"] = str(workspace_path)

    # Save config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return workspace_path


def run_workflow(
    user_workspace: str | Path | None = None,
    config: str | dict[str, Any] | None = None,
    targets: list[str] | None = None,
    cores: int = 1,
    dryrun: bool = False,
    forceall: bool = False,
    forcetargets: bool = False,
    **snakemake_kwargs,
) -> bool:
    """
    Execute PyPSA-USA workflow programmatically.

    Args:
        user_workspace: Path to directory where all intermediate files, results,
                       and logs will be stored. If None, uses the default workspace
                       from user configuration. This directory will be created if
                       it doesn't exist. Should be an absolute path for clarity.
        config: Path to config file or config dictionary
        targets: List of target files/rules to build
        cores: Number of CPU cores to use
        dryrun: If True, only show what would be done
        forceall: Force execution of all rules
        forcetargets: Force execution of target rules
        **snakemake_kwargs: Additional arguments passed to snakemake

    Returns
    -------
        True if workflow completed successfully, False otherwise

    Example:
        >>> from pypsa_usa.api import run_workflow, set_default_workspace
        >>> from pathlib import Path
        >>>
        >>> # Set your default workspace (only needed once)
        >>> set_default_workspace("/path/to/my/project/workspace")
        >>>
        >>> # Now you can run workflows without specifying workspace
        >>> success = run_workflow(targets=["all"], cores=4)
        >>>
        >>> # Or override the default for specific runs
        >>> success = run_workflow(
        ...     user_workspace="/home/user/special_project",
        ...     config="my_config.yaml",
        ...     targets=["results/western/figures/"],
        ...     cores=8,
        ... )
    """
    # Resolve workspace path
    if user_workspace is None:
        # Try to get default workspace from user config
        default_workspace = get_default_workspace()
        if default_workspace is None:
            raise ValueError(
                "No user_workspace provided and no default workspace set. "
                "Please either:\n"
                "1. Provide user_workspace parameter, or\n"
                "2. Set a default workspace using set_default_workspace()",
            )
        workspace_path = default_workspace
    else:
        workspace_path = Path(user_workspace).expanduser().resolve()

    # Ensure workspace directory exists
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Get workflow path
    workflow_path = get_workflow_path()
    snakefile = workflow_path / "Snakefile"

    # Set default targets
    if targets is None:
        targets = ["all"]

    # Prepare snakemake arguments
    snakemake_args = {
        "snakefile": str(snakefile),
        "workdir": str(workspace_path),
        "cores": cores,
        "targets": targets,
        "dryrun": dryrun,
        "forceall": forceall,
        "forcetargets": forcetargets,
        **snakemake_kwargs,
    }

    # Handle config
    if config is not None:
        if isinstance(config, str):
            # Config file path
            config_path = Path(config)
            if not config_path.is_absolute():
                config_path = workspace_path / "config" / config
            snakemake_args["configfiles"] = [str(config_path)]
        elif isinstance(config, dict):
            # Config dictionary
            snakemake_args["config"] = config

    try:
        # Set environment variables to ensure Snakemake uses the correct Python interpreter
        # This is important for script execution in rules that don't have conda environments
        original_python = os.environ.get("PYTHON", None)
        original_pythonpath = os.environ.get("PYTHONPATH", None)

        # Set the Python interpreter to the current one (which should be from uv environment)
        os.environ["PYTHON"] = sys.executable

        # Add the current Python path to PYTHONPATH to ensure modules are found
        current_pythonpath = os.pathsep.join(sys.path)
        if original_pythonpath:
            os.environ["PYTHONPATH"] = current_pythonpath + os.pathsep + original_pythonpath
        else:
            os.environ["PYTHONPATH"] = current_pythonpath

        try:
            # Execute workflow
            success = snakemake.snakemake(**snakemake_args)
            return success
        finally:
            # Restore original environment variables
            if original_python is not None:
                os.environ["PYTHON"] = original_python
            elif "PYTHON" in os.environ:
                del os.environ["PYTHON"]

            if original_pythonpath is not None:
                os.environ["PYTHONPATH"] = original_pythonpath
            elif "PYTHONPATH" in os.environ:
                del os.environ["PYTHONPATH"]

    except Exception as e:
        print(f"Error running workflow: {e}")
        return False


def create_user_config(
    user_workspace: str | Path,
    config_name: str = "config.default.yaml",
    template: str | None = None,
) -> Path:
    """
    Create a user config file from template.

    Args:
        user_workspace: Path to user workspace directory
        config_name: Name of the config file to create
        template: Template config name (defaults to config.default.yaml)

    Returns
    -------
        Path to the created config file

    Example:
        >>> from pypsa_usa.api import create_user_config
        >>> from pathlib import Path
        >>>
        >>> # Create default config
        >>> workspace = Path("/path/to/my/workspace")
        >>> config_path = create_user_config(workspace, "my_project.yaml")
        >>>
        >>> # Create config from specific template
        >>> config_path = create_user_config(
        ...     "/home/user/pypsa_project",
        ...     "my_project.yaml",
        ...     template="config.tutorial.yaml",
        ... )
    """
    import shutil

    # Set default template
    if template is None:
        template = "config.default.yaml"

    # Convert user_workspace to Path
    workspace_path = Path(user_workspace).resolve()

    # Get template path
    template_path = get_workflow_path() / "config" / template

    # Get user config path
    user_config_dir = workspace_path / "config"
    user_config_dir.mkdir(parents=True, exist_ok=True)
    user_config_path = user_config_dir / config_name

    # Copy template to user config
    shutil.copy2(template_path, user_config_path)

    return user_config_path


def list_available_configs() -> list[str]:
    """
    List available config templates.

    Returns
    -------
        List of available config template names
    """
    config_dir = get_workflow_path() / "config"
    config_files = list(config_dir.glob("*.yaml"))
    return [f.name for f in config_files]


def get_workflow_info() -> dict[str, Any]:
    """
    Get information about the PyPSA-USA workflow.

    Returns
    -------
        Dictionary with workflow information
    """
    workflow_path = get_workflow_path()

    return {
        "workflow_path": str(workflow_path),
        "snakefile": str(workflow_path / "Snakefile"),
        "config_dir": str(workflow_path / "config"),
        "rules_dir": str(workflow_path / "rules"),
        "scripts_dir": str(workflow_path / "scripts"),
        "available_configs": list_available_configs(),
    }


def setup_user_workspace(user_workspace: str | Path) -> Path:
    """
    Set up user workspace directory structure.

    Args:
        user_workspace: Path to user workspace directory

    Returns
    -------
        Path to the user workspace directory
    """
    workspace = Path(user_workspace).resolve()

    # Create subdirectories
    subdirs = ["my_configs", "data", "resources", "results", "logs", "benchmarks", "cutouts"]
    for subdir in subdirs:
        (workspace / subdir).mkdir(parents=True, exist_ok=True)

    return workspace


def touch(
    user_workspace: str | Path | None = None,
    config: str | dict[str, Any] | None = None,
    targets: list[str] | None = None,
    cores: int = 1,
    **snakemake_kwargs,
) -> bool:
    """
    Touch output files in PyPSA-USA workflow without executing rules.

    This function updates the timestamps of output files to mark them as up-to-date
    without actually running the workflow rules. This is useful for testing or
    when you want to mark files as current without the computational cost.

    Args:
        user_workspace: Path to directory where all intermediate files, results,
                       and logs will be stored. If None, uses the default workspace
                       from user configuration. This directory will be created if
                       it doesn't exist. Should be an absolute path for clarity.
        config: Path to config file or config dictionary
        targets: List of target files/rules to touch
        cores: Number of CPU cores to use (for snakemake compatibility)
        **snakemake_kwargs: Additional arguments passed to snakemake

    Returns
    -------
        True if touch operation completed successfully, False otherwise

    Example:
        >>> from pypsa_usa.api import touch, set_default_workspace
        >>> from pathlib import Path
        >>>
        >>> # Set your default workspace (only needed once)
        >>> set_default_workspace("/path/to/my/project/workspace")
        >>>
        >>> # Touch all targets without running them
        >>> success = touch(targets=["all"])
        >>>
        >>> # Touch specific targets with custom config
        >>> success = touch(
        ...     user_workspace="/home/user/special_project",
        ...     config="my_config.yaml",
        ...     targets=["results/western/figures/"],
        ... )
    """
    # Resolve workspace path
    if user_workspace is None:
        # Try to get default workspace from user config
        default_workspace = get_default_workspace()
        if default_workspace is None:
            raise ValueError(
                "No user_workspace provided and no default workspace set. "
                "Please either:\n"
                "1. Provide user_workspace parameter, or\n"
                "2. Set a default workspace using set_default_workspace()",
            )
        workspace_path = default_workspace
    else:
        workspace_path = Path(user_workspace).expanduser().resolve()

    # Ensure workspace directory exists
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Get workflow path
    workflow_path = get_workflow_path()
    snakefile = workflow_path / "Snakefile"

    until = snakemake_kwargs.get("until", None)

    # Set default targets
    if targets is None:
        targets = ["all"]

    # Prepare snakemake arguments with touch flag
    snakemake_args = {
        "snakefile": str(snakefile),
        "workdir": str(workspace_path),
        "cores": cores,
        "targets": targets,
        "touch": True,  # This is the key flag for touch functionality
        **snakemake_kwargs,
        "until": until,
    }

    # Handle config
    if config is not None:
        if isinstance(config, str):
            # Config file path
            config_path = Path(config)
            if not config_path.is_absolute():
                config_path = workspace_path / "config" / config
            snakemake_args["configfiles"] = [str(config_path)]
        elif isinstance(config, dict):
            # Config dictionary
            snakemake_args["config"] = config

    try:
        # Set environment variables to ensure Snakemake uses the correct Python interpreter
        # This is important for script execution in rules that don't have conda environments
        original_python = os.environ.get("PYTHON", None)
        original_pythonpath = os.environ.get("PYTHONPATH", None)

        # Set the Python interpreter to the current one (which should be from uv environment)
        os.environ["PYTHON"] = sys.executable

        # Add the current Python path to PYTHONPATH to ensure modules are found
        current_pythonpath = os.pathsep.join(sys.path)
        if original_pythonpath:
            os.environ["PYTHONPATH"] = current_pythonpath + os.pathsep + original_pythonpath
        else:
            os.environ["PYTHONPATH"] = current_pythonpath

        try:
            # Execute touch operation
            success = snakemake.snakemake(**snakemake_args)
            return success
        finally:
            # Restore original environment variables
            if original_python is not None:
                os.environ["PYTHON"] = original_python
            elif "PYTHON" in os.environ:
                del os.environ["PYTHON"]

            if original_pythonpath is not None:
                os.environ["PYTHONPATH"] = original_pythonpath
            elif "PYTHONPATH" in os.environ:
                del os.environ["PYTHONPATH"]

    except Exception as e:
        print(f"Error touching workflow files: {e}")
        return False
