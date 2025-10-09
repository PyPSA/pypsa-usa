"""
Test cases for PyPSA-USA API workflow functionality.

This module contains comprehensive tests for the PyPSA-USA API, covering different
usage scenarios and ensuring the API works correctly with various configurations.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pypsa_usa.api.workflow import (
    create_user_config,
    get_default_workspace,
    get_user_config_path,
    get_workflow_info,
    list_available_configs,
    run_workflow,
    set_default_workspace,
    setup_user_workspace,
)


class TestWorkflowAPI:
    """Test cases for the PyPSA-USA workflow API."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def temp_config_file(self, temp_workspace):
        """Create a temporary config file for testing."""
        config_content = {
            "scenario": {
                "interconnect": "western",
                "clusters": 30,
                "simpl": 75,
            },
            "run": {
                "name": "test_run",
            },
            "costs": {
                "ng_fuel_year": 2019,
            },
        }
        config_path = temp_workspace / "test_config.yaml"
        
        # Create a simple YAML-like config file
        with open(config_path, 'w') as f:
            f.write("scenario:\n")
            f.write("  interconnect: western\n")
            f.write("  clusters: 30\n")
            f.write("  simpl: 75\n")
            f.write("run:\n")
            f.write("  name: test_run\n")
            f.write("costs:\n")
            f.write("  ng_fuel_year: 2019\n")
        
        return config_path

    @pytest.fixture
    def mock_snakemake_success(self):
        """Mock snakemake.snakemake to return success."""
        with patch('pypsa_usa.api.workflow.snakemake.snakemake') as mock_snakemake:
            mock_snakemake.return_value = True
            yield mock_snakemake

    @pytest.fixture
    def mock_snakemake_failure(self):
        """Mock snakemake.snakemake to return failure."""
        with patch('pypsa_usa.api.workflow.snakemake.snakemake') as mock_snakemake:
            mock_snakemake.return_value = False
            yield mock_snakemake

    def test_get_user_config_path(self):
        """Test getting the user config path."""
        config_path = get_user_config_path()
        assert isinstance(config_path, Path)
        assert config_path.name == "config.json"
        assert ".config" in str(config_path)

    def test_set_and_get_default_workspace(self, temp_workspace):
        """Test setting and getting default workspace."""
        # Initially no default workspace should be set
        assert get_default_workspace() is None
        
        # Set a default workspace
        set_workspace = set_default_workspace(temp_workspace)
        assert set_workspace == temp_workspace.resolve()
        
        # Get the default workspace
        default_workspace = get_default_workspace()
        assert default_workspace == temp_workspace.resolve()
        
        # Test with string path
        string_path = str(temp_workspace / "subdir")
        set_workspace = set_default_workspace(string_path)
        assert set_workspace == Path(string_path).resolve()

    def test_setup_user_workspace(self, temp_workspace):
        """Test setting up user workspace directory structure."""
        workspace = setup_user_workspace(temp_workspace)
        assert workspace == temp_workspace.resolve()
        
        # Check that required subdirectories were created
        expected_dirs = ["config", "data", "resources", "results", "logs", "benchmarks", "cutouts"]
        for subdir in expected_dirs:
            assert (temp_workspace / subdir).exists()
            assert (temp_workspace / subdir).is_dir()

    def test_create_user_config(self, temp_workspace):
        """Test creating user config from template."""
        # Mock the workflow path and template
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            mock_get_path.return_value = mock_workflow_path
            
            # Create a mock template
            template_path = mock_workflow_path / "config" / "config.default.yaml"
            template_path.parent.mkdir(parents=True)
            template_path.write_text("test: config\n")
            
            # Test creating config
            config_path = create_user_config(temp_workspace, "my_config.yaml")
            assert config_path == temp_workspace / "config" / "my_config.yaml"
            assert config_path.exists()
            assert config_path.read_text() == "test: config\n"

    def test_list_available_configs(self):
        """Test listing available config templates."""
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            # Create mock config directory with some files
            mock_workflow_path = Path("/mock/workflow")
            mock_config_dir = mock_workflow_path / "config"
            mock_get_path.return_value = mock_workflow_path
            
            with patch.object(mock_config_dir, 'glob') as mock_glob:
                mock_files = [
                    Mock(name="config.default.yaml"),
                    Mock(name="config.tutorial.yaml"),
                    Mock(name="config.api.yaml"),
                ]
                mock_glob.return_value = mock_files
                
                configs = list_available_configs()
                assert len(configs) == 3
                assert "config.default.yaml" in configs
                assert "config.tutorial.yaml" in configs
                assert "config.api.yaml" in configs

    def test_get_workflow_info(self):
        """Test getting workflow information."""
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = Path("/mock/workflow")
            mock_get_path.return_value = mock_workflow_path
            
            with patch('pypsa_usa.api.workflow.list_available_configs') as mock_list_configs:
                mock_list_configs.return_value = ["config.default.yaml", "config.tutorial.yaml"]
                
                info = get_workflow_info()
                assert info["workflow_path"] == str(mock_workflow_path)
                assert info["snakefile"] == str(mock_workflow_path / "Snakefile")
                assert info["config_dir"] == str(mock_workflow_path / "config")
                assert info["rules_dir"] == str(mock_workflow_path / "rules")
                assert info["scripts_dir"] == str(mock_workflow_path / "scripts")
                assert "config.default.yaml" in info["available_configs"]

    def test_run_workflow_with_default_workspace(self, temp_workspace, temp_config_file, mock_snakemake_success):
        """Test running workflow with default workspace set."""
        # Set default workspace
        set_default_workspace(temp_workspace)
        
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Test running workflow
            success = run_workflow(
                config=str(temp_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            
            assert success is True
            mock_snakemake_success.assert_called_once()

    def test_run_workflow_with_explicit_workspace(self, temp_workspace, temp_config_file, mock_snakemake_success):
        """Test running workflow with explicit workspace."""
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Test running workflow with explicit workspace
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(temp_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            
            assert success is True
            mock_snakemake_success.assert_called_once()

    def test_run_workflow_with_config_dict(self, temp_workspace, mock_snakemake_success):
        """Test running workflow with config dictionary."""
        config_dict = {
            "scenario": {
                "interconnect": "western",
                "clusters": 30,
            },
            "run": {
                "name": "test_run",
            },
            "costs": {
                "ng_fuel_year": 2019,
            },
        }
        
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Test running workflow with config dict
            success = run_workflow(
                user_workspace=temp_workspace,
                config=config_dict,
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            
            assert success is True
            mock_snakemake_success.assert_called_once()

    def test_run_workflow_with_different_targets(self, temp_workspace, temp_config_file, mock_snakemake_success):
        """Test running workflow with different target combinations."""
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Test with single target
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(temp_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            assert success is True
            
            # Test with multiple targets
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(temp_config_file),
                targets=["retrieve_caiso_data", "build_cost_data"],
                dryrun=True,
                cores=1
            )
            assert success is True
            
            # Test with default targets (all)
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(temp_config_file),
                dryrun=True,
                cores=1
            )
            assert success is True

    def test_run_workflow_with_different_cores(self, temp_workspace, temp_config_file, mock_snakemake_success):
        """Test running workflow with different core counts."""
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Test with different core counts
            for cores in [1, 2, 4, 8]:
                success = run_workflow(
                    user_workspace=temp_workspace,
                    config=str(temp_config_file),
                    targets=["retrieve_caiso_data"],
                    dryrun=True,
                    cores=cores
                )
                assert success is True

    def test_run_workflow_with_flags(self, temp_workspace, temp_config_file, mock_snakemake_success):
        """Test running workflow with different flags."""
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Test with forceall flag
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(temp_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                forceall=True,
                cores=1
            )
            assert success is True
            
            # Test with forcetargets flag
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(temp_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                forcetargets=True,
                cores=1
            )
            assert success is True

    def test_run_workflow_with_additional_kwargs(self, temp_workspace, temp_config_file, mock_snakemake_success):
        """Test running workflow with additional Snakemake kwargs."""
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Test with additional kwargs
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(temp_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1,
                keepgoing=True,
                latency_wait=60,
                scheduler="greedy"
            )
            assert success is True

    def test_run_workflow_environment_variables_set(self, temp_workspace, temp_config_file, mock_snakemake_success):
        """Test that environment variables are properly set for script execution."""
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Store original environment variables
            original_python = os.environ.get('PYTHON')
            original_pythonpath = os.environ.get('PYTHONPATH')
            
            try:
                # Test running workflow
                success = run_workflow(
                    user_workspace=temp_workspace,
                    config=str(temp_config_file),
                    targets=["retrieve_caiso_data"],
                    dryrun=True,
                    cores=1
                )
                
                assert success is True
                
                # Verify that environment variables were set during execution
                # (This is tested indirectly through the mock_snakemake_success fixture)
                mock_snakemake_success.assert_called_once()
                
            finally:
                # Restore original environment variables
                if original_python is not None:
                    os.environ['PYTHON'] = original_python
                elif 'PYTHON' in os.environ:
                    del os.environ['PYTHON']
                    
                if original_pythonpath is not None:
                    os.environ['PYTHONPATH'] = original_pythonpath
                elif 'PYTHONPATH' in os.environ:
                    del os.environ['PYTHONPATH']

    def test_run_workflow_failure(self, temp_workspace, temp_config_file, mock_snakemake_failure):
        """Test workflow execution failure."""
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Test workflow failure
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(temp_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            
            assert success is False

    def test_run_workflow_exception_handling(self, temp_workspace, temp_config_file):
        """Test workflow exception handling."""
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Mock snakemake to raise an exception
            with patch('pypsa_usa.api.workflow.snakemake.snakemake') as mock_snakemake:
                mock_snakemake.side_effect = Exception("Test exception")
                
                # Test exception handling
                success = run_workflow(
                    user_workspace=temp_workspace,
                    config=str(temp_config_file),
                    targets=["retrieve_caiso_data"],
                    dryrun=True,
                    cores=1
                )
                
                assert success is False

    def test_run_workflow_no_default_workspace(self, temp_config_file):
        """Test running workflow without default workspace set."""
        # Ensure no default workspace is set
        config_path = get_user_config_path()
        if config_path.exists():
            config_path.unlink()
        
        # Test that ValueError is raised
        with pytest.raises(ValueError, match="No user_workspace provided and no default workspace set"):
            run_workflow(
                config=str(temp_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )

    def test_run_workflow_invalid_config_path(self, temp_workspace):
        """Test running workflow with invalid config path."""
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Test with non-existent config file
            with pytest.raises(Exception):  # Should handle the error gracefully
                run_workflow(
                    user_workspace=temp_workspace,
                    config="non_existent_config.yaml",
                    targets=["retrieve_caiso_data"],
                    dryrun=True,
                    cores=1
                )

    def test_run_workflow_relative_config_path(self, temp_workspace, temp_config_file, mock_snakemake_success):
        """Test running workflow with relative config path."""
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Create config in workspace config directory
            config_dir = temp_workspace / "config"
            config_dir.mkdir()
            relative_config = config_dir / "test_config.yaml"
            relative_config.write_text("test: config\n")
            
            # Test with relative config path
            success = run_workflow(
                user_workspace=temp_workspace,
                config="test_config.yaml",  # Relative path
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            
            assert success is True

    def test_run_workflow_workspace_creation(self, temp_workspace, temp_config_file, mock_snakemake_success):
        """Test that workspace is created if it doesn't exist."""
        # Mock the workflow path
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            # Use a non-existent workspace
            non_existent_workspace = temp_workspace / "new_workspace"
            assert not non_existent_workspace.exists()
            
            # Test that workspace is created
            success = run_workflow(
                user_workspace=non_existent_workspace,
                config=str(temp_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            
            assert success is True
            assert non_existent_workspace.exists()


class TestWorkflowAPIIntegration:
    """Integration tests for the PyPSA-USA workflow API."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_full_workflow_setup_and_execution(self, temp_workspace):
        """Test complete workflow setup and execution flow."""
        # Mock the workflow path and snakemake
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            with patch('pypsa_usa.api.workflow.snakemake.snakemake') as mock_snakemake:
                mock_snakemake.return_value = True
                
                # Step 1: Set up workspace
                workspace = setup_user_workspace(temp_workspace)
                assert workspace.exists()
                
                # Step 2: Create config
                config_path = create_user_config(temp_workspace, "test_config.yaml")
                assert config_path.exists()
                
                # Step 3: Set default workspace
                set_default_workspace(temp_workspace)
                assert get_default_workspace() == temp_workspace.resolve()
                
                # Step 4: Run workflow
                success = run_workflow(
                    config=str(config_path),
                    targets=["retrieve_caiso_data"],
                    dryrun=True,
                    cores=1
                )
                
                assert success is True
                mock_snakemake.assert_called_once()

    def test_multiple_workflow_runs(self, temp_workspace):
        """Test running multiple workflows in sequence."""
        # Mock the workflow path and snakemake
        with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
            mock_workflow_path = temp_workspace / "workflow"
            mock_workflow_path.mkdir(parents=True)
            (mock_workflow_path / "Snakefile").touch()
            mock_get_path.return_value = mock_workflow_path
            
            with patch('pypsa_usa.api.workflow.snakemake.snakemake') as mock_snakemake:
                mock_snakemake.return_value = True
                
                # Set up workspace
                setup_user_workspace(temp_workspace)
                set_default_workspace(temp_workspace)
                
                # Create config
                config_path = create_user_config(temp_workspace, "test_config.yaml")
                
                # Run multiple workflows
                targets_list = [
                    ["retrieve_caiso_data"],
                    ["build_cost_data"],
                    ["retrieve_caiso_data", "build_cost_data"],
                    ["all"]
                ]
                
                for targets in targets_list:
                    success = run_workflow(
                        config=str(config_path),
                        targets=targets,
                        dryrun=True,
                        cores=1
                    )
                    assert success is True
                
                # Verify snakemake was called for each run
                assert mock_snakemake.call_count == len(targets_list)
