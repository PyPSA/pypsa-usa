"""
Test cases for different PyPSA-USA API usage scenarios.

This module tests the various ways users can interact with the PyPSA-USA API,
covering common usage patterns and edge cases.
"""

from pypsa_usa.api.workflow import run_workflow, set_default_workspace


class TestWorkflowUsageScenarios:
    """Test different ways users can run PyPSA-USA workflows."""

    def test_basic_usage_with_default_workspace(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test basic usage pattern: set default workspace, then run workflow."""
        # Set default workspace (typical first-time setup)
        set_default_workspace(temp_workspace)

        # Run workflow without specifying workspace (uses default)
        success = run_workflow(
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )

        assert success is True
        mock_snakemake_success.assert_called_once()

    def test_usage_with_explicit_workspace(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with explicit workspace specification."""
        # Run workflow with explicit workspace (no default needed)
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )

        assert success is True
        mock_snakemake_success.assert_called_once()

    def test_usage_with_config_dictionary(
        self,
        temp_workspace,
        sample_config_dict,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with configuration dictionary instead of file."""
        # Run workflow with config dictionary
        success = run_workflow(
            user_workspace=temp_workspace,
            config=sample_config_dict,
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )

        assert success is True
        mock_snakemake_success.assert_called_once()

    def test_usage_with_relative_config_path(self, temp_workspace, mock_workflow_path, mock_snakemake_success):
        """Test usage with relative config path."""
        # Create config in workspace config directory
        config_dir = temp_workspace / "config"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "my_config.yaml"
        config_file.write_text("scenario:\n  interconnect: western\n")

        # Run workflow with relative config path
        success = run_workflow(
            user_workspace=temp_workspace,
            config="my_config.yaml",  # Relative path
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )

        assert success is True
        mock_snakemake_success.assert_called_once()

    def test_usage_with_multiple_targets(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with multiple targets."""
        # Run workflow with multiple targets
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data", "build_cost_data", "build_powerplants"],
            dryrun=True,
            cores=1,
        )

        assert success is True
        mock_snakemake_success.assert_called_once()

    def test_usage_with_default_targets(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with default targets (all)."""
        # Run workflow without specifying targets (uses default "all")
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            dryrun=True,
            cores=1,
        )

        assert success is True
        mock_snakemake_success.assert_called_once()

    def test_usage_with_different_core_counts(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with different core counts."""
        # Test with different core counts
        for cores in [1, 2, 4, 8, 16]:
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(sample_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=cores,
            )
            assert success is True

    def test_usage_with_force_flags(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with force flags."""
        # Test with forceall flag
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            forceall=True,
            cores=1,
        )
        assert success is True

        # Test with forcetargets flag
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            forcetargets=True,
            cores=1,
        )
        assert success is True

    def test_usage_with_additional_snakemake_kwargs(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with additional Snakemake keyword arguments."""
        # Run workflow with additional kwargs
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
            keepgoing=True,
            latency_wait=60,
            scheduler="greedy",
            printshellcmds=True,
        )

        assert success is True
        mock_snakemake_success.assert_called_once()

    def test_usage_with_dryrun_vs_execution(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with dryrun vs actual execution."""
        # Test dryrun
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )
        assert success is True

        # Test actual execution (dryrun=False)
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=False,
            cores=1,
        )
        assert success is True

    def test_usage_with_string_vs_path_workspace(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with string vs Path object for workspace."""
        # Test with string workspace
        success = run_workflow(
            user_workspace=str(temp_workspace),
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )
        assert success is True

        # Test with Path object workspace
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )
        assert success is True

    def test_usage_with_string_vs_path_config(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with string vs Path object for config."""
        # Test with string config path
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )
        assert success is True

        # Test with Path object config path
        success = run_workflow(
            user_workspace=temp_workspace,
            config=sample_config_file,
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )
        assert success is True

    def test_usage_with_expanded_workspace_path(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with expanded workspace path (e.g., with ~)."""
        # Create a subdirectory to test path expansion
        subdir = temp_workspace / "subdir"
        subdir.mkdir()

        # Test with relative path that needs expansion
        success = run_workflow(
            user_workspace=subdir,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )
        assert success is True

    def test_usage_with_nonexistent_workspace_creation(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage where workspace doesn't exist and gets created."""
        # Use a non-existent workspace path
        nonexistent_workspace = temp_workspace / "new_workspace"
        assert not nonexistent_workspace.exists()

        # Run workflow - should create the workspace
        success = run_workflow(
            user_workspace=nonexistent_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )

        assert success is True
        assert nonexistent_workspace.exists()

    def test_usage_with_complex_config_dict(self, temp_workspace, mock_workflow_path, mock_snakemake_success):
        """Test usage with complex configuration dictionary."""
        complex_config = {
            "scenario": {
                "interconnect": "western",
                "clusters": 50,
                "simpl": 100,
            },
            "run": {
                "name": "complex_test_run",
                "sector": "E",
            },
            "costs": {
                "ng_fuel_year": 2020,
                "year": 2030,
            },
            "solving": {
                "solver": "gurobi",
                "options": {
                    "threads": 4,
                    "time_limit": 3600,
                },
            },
            "plotting": {
                "map": True,
                "statistics": True,
            },
        }

        # Run workflow with complex config
        success = run_workflow(
            user_workspace=temp_workspace,
            config=complex_config,
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )

        assert success is True
        mock_snakemake_success.assert_called_once()

    def test_usage_with_specific_rule_targets(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with specific rule targets."""
        # Test with various specific rule targets
        rule_targets = [
            "retrieve_caiso_data",
            "build_cost_data",
            "build_powerplants",
            "build_shapes",
            "build_base_network",
            "cluster_network",
            "solve_network",
            "plot_network_maps",
            "plot_statistics",
        ]

        for target in rule_targets:
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(sample_config_file),
                targets=[target],
                dryrun=True,
                cores=1,
            )
            assert success is True

    def test_usage_with_mixed_target_types(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with mixed target types (rules and files)."""
        # Test with mixed targets (rules and potential file targets)
        mixed_targets = [
            "retrieve_caiso_data",  # Rule
            "build_cost_data",  # Rule
            "data/costs/caiso_ng_power_prices.csv",  # File target
            "resources/powerplants.csv",  # File target
        ]

        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=mixed_targets,
            dryrun=True,
            cores=1,
        )

        assert success is True
        mock_snakemake_success.assert_called_once()

    def test_usage_with_environment_variables_preserved(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
        clean_environment,
    ):
        """Test that environment variables are properly preserved."""
        import os

        # Set some test environment variables
        os.environ["TEST_VAR"] = "test_value"
        os.environ["PYTHON"] = "/usr/bin/python3"

        # Run workflow
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )

        assert success is True

        # Verify environment variables are still set
        assert os.environ.get("TEST_VAR") == "test_value"
        assert os.environ.get("PYTHON") == "/usr/bin/python3"

    def test_usage_with_concurrent_workflows(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_success,
    ):
        """Test usage with multiple concurrent workflow runs."""
        # This tests that the API can handle multiple calls
        # (though in practice they would be sequential due to GIL)

        workspaces = [temp_workspace / f"workspace_{i}" for i in range(3)]

        for workspace in workspaces:
            success = run_workflow(
                user_workspace=workspace,
                config=str(sample_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1,
            )
            assert success is True
            assert workspace.exists()

    def test_usage_with_error_handling(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_failure,
    ):
        """Test usage with error handling scenarios."""
        # Test with snakemake failure
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )

        assert success is False

    def test_usage_with_exception_handling(
        self,
        temp_workspace,
        sample_config_file,
        mock_workflow_path,
        mock_snakemake_exception,
    ):
        """Test usage with exception handling scenarios."""
        # Test with snakemake exception
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1,
        )

        assert success is False
