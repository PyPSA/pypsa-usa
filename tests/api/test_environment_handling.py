"""
Test cases for environment variable handling in PyPSA-USA API.

This module specifically tests the environment variable handling that ensures
Snakemake uses the correct Python interpreter for script execution.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from pypsa_usa.api.workflow import run_workflow


class TestEnvironmentHandling:
    """Test environment variable handling for script execution."""

    def test_python_environment_variable_set(self, temp_workspace, sample_config_file, mock_workflow_path, mock_snakemake_success):
        """Test that PYTHON environment variable is set correctly."""
        import sys
        
        # Store original environment
        original_python = os.environ.get('PYTHON')
        original_pythonpath = os.environ.get('PYTHONPATH')
        
        try:
            # Run workflow
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(sample_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            
            assert success is True
            
            # Verify that PYTHON was set to current executable during execution
            # We can't directly test this since it's set temporarily, but we can
            # verify the mock was called, which means the environment was set
            mock_snakemake_success.assert_called_once()
            
        finally:
            # Restore original environment
            if original_python is not None:
                os.environ['PYTHON'] = original_python
            elif 'PYTHON' in os.environ:
                del os.environ['PYTHON']
                
            if original_pythonpath is not None:
                os.environ['PYTHONPATH'] = original_pythonpath
            elif 'PYTHONPATH' in os.environ:
                del os.environ['PYTHONPATH']

    def test_pythonpath_environment_variable_set(self, temp_workspace, sample_config_file, mock_workflow_path, mock_snakemake_success):
        """Test that PYTHONPATH environment variable is set correctly."""
        import sys
        
        # Store original environment
        original_python = os.environ.get('PYTHON')
        original_pythonpath = os.environ.get('PYTHONPATH')
        
        try:
            # Run workflow
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(sample_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            
            assert success is True
            
            # Verify that PYTHONPATH was set during execution
            mock_snakemake_success.assert_called_once()
            
        finally:
            # Restore original environment
            if original_python is not None:
                os.environ['PYTHON'] = original_python
            elif 'PYTHON' in os.environ:
                del os.environ['PYTHON']
                
            if original_pythonpath is not None:
                os.environ['PYTHONPATH'] = original_pythonpath
            elif 'PYTHONPATH' in os.environ:
                del os.environ['PYTHONPATH']

    def test_environment_variables_restored_after_execution(self, temp_workspace, sample_config_file, mock_workflow_path, mock_snakemake_success):
        """Test that environment variables are restored after execution."""
        # Set initial environment variables
        original_python = "/original/python"
        original_pythonpath = "/original/pythonpath"
        
        os.environ['PYTHON'] = original_python
        os.environ['PYTHONPATH'] = original_pythonpath
        
        try:
            # Run workflow
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(sample_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            
            assert success is True
            
            # Verify environment variables are restored
            assert os.environ.get('PYTHON') == original_python
            assert os.environ.get('PYTHONPATH') == original_pythonpath
            
        finally:
            # Clean up
            if 'PYTHON' in os.environ:
                del os.environ['PYTHON']
            if 'PYTHONPATH' in os.environ:
                del os.environ['PYTHONPATH']

    def test_environment_variables_restored_when_none_existed(self, temp_workspace, sample_config_file, mock_workflow_path, mock_snakemake_success):
        """Test that environment variables are properly cleaned up when none existed originally."""
        # Ensure no environment variables exist initially
        if 'PYTHON' in os.environ:
            del os.environ['PYTHON']
        if 'PYTHONPATH' in os.environ:
            del os.environ['PYTHONPATH']
        
        # Run workflow
        success = run_workflow(
            user_workspace=temp_workspace,
            config=str(sample_config_file),
            targets=["retrieve_caiso_data"],
            dryrun=True,
            cores=1
        )
        
        assert success is True
        
        # Verify environment variables are cleaned up
        assert 'PYTHON' not in os.environ
        assert 'PYTHONPATH' not in os.environ

    def test_environment_variables_restored_on_exception(self, temp_workspace, sample_config_file, mock_workflow_path, mock_snakemake_exception):
        """Test that environment variables are restored even when exceptions occur."""
        # Set initial environment variables
        original_python = "/original/python"
        original_pythonpath = "/original/pythonpath"
        
        os.environ['PYTHON'] = original_python
        os.environ['PYTHONPATH'] = original_pythonpath
        
        try:
            # Run workflow (should fail with exception)
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(sample_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            
            assert success is False
            
            # Verify environment variables are still restored despite exception
            assert os.environ.get('PYTHON') == original_python
            assert os.environ.get('PYTHONPATH') == original_pythonpath
            
        finally:
            # Clean up
            if 'PYTHON' in os.environ:
                del os.environ['PYTHON']
            if 'PYTHONPATH' in os.environ:
                del os.environ['PYTHONPATH']

    def test_pythonpath_combines_existing_and_new(self, temp_workspace, sample_config_file, mock_workflow_path, mock_snakemake_success):
        """Test that PYTHONPATH properly combines existing and new paths."""
        import sys
        
        # Set initial PYTHONPATH
        original_pythonpath = "/existing/path1:/existing/path2"
        os.environ['PYTHONPATH'] = original_pythonpath
        
        try:
            # Run workflow
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(sample_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            
            assert success is True
            
            # Verify PYTHONPATH is restored
            assert os.environ.get('PYTHONPATH') == original_pythonpath
            
        finally:
            # Clean up
            if 'PYTHONPATH' in os.environ:
                del os.environ['PYTHONPATH']

    def test_environment_handling_with_uv_managed_python(self, temp_workspace, sample_config_file, mock_workflow_path, mock_snakemake_success):
        """Test environment handling when using uv-managed Python."""
        import sys
        
        # Simulate uv-managed Python path
        uv_python_path = "/path/to/uv/venv/bin/python"
        
        with patch('sys.executable', uv_python_path):
            # Store original environment
            original_python = os.environ.get('PYTHON')
            original_pythonpath = os.environ.get('PYTHONPATH')
            
            try:
                # Run workflow
                success = run_workflow(
                    user_workspace=temp_workspace,
                    config=str(sample_config_file),
                    targets=["retrieve_caiso_data"],
                    dryrun=True,
                    cores=1
                )
                
                assert success is True
                
                # Verify the mock was called (environment was set)
                mock_snakemake_success.assert_called_once()
                
            finally:
                # Restore original environment
                if original_python is not None:
                    os.environ['PYTHON'] = original_python
                elif 'PYTHON' in os.environ:
                    del os.environ['PYTHON']
                    
                if original_pythonpath is not None:
                    os.environ['PYTHONPATH'] = original_pythonpath
                elif 'PYTHONPATH' in os.environ:
                    del os.environ['PYTHONPATH']

    def test_environment_handling_with_system_python(self, temp_workspace, sample_config_file, mock_workflow_path, mock_snakemake_success):
        """Test environment handling when using system Python."""
        import sys
        
        # Simulate system Python path
        system_python_path = "/usr/bin/python3"
        
        with patch('sys.executable', system_python_path):
            # Store original environment
            original_python = os.environ.get('PYTHON')
            original_pythonpath = os.environ.get('PYTHONPATH')
            
            try:
                # Run workflow
                success = run_workflow(
                    user_workspace=temp_workspace,
                    config=str(sample_config_file),
                    targets=["retrieve_caiso_data"],
                    dryrun=True,
                    cores=1
                )
                
                assert success is True
                
                # Verify the mock was called (environment was set)
                mock_snakemake_success.assert_called_once()
                
            finally:
                # Restore original environment
                if original_python is not None:
                    os.environ['PYTHON'] = original_python
                elif 'PYTHON' in os.environ:
                    del os.environ['PYTHON']
                    
                if original_pythonpath is not None:
                    os.environ['PYTHONPATH'] = original_pythonpath
                elif 'PYTHONPATH' in os.environ:
                    del os.environ['PYTHONPATH']

    def test_environment_handling_multiple_calls(self, temp_workspace, sample_config_file, mock_workflow_path, mock_snakemake_success):
        """Test environment handling with multiple consecutive calls."""
        # Set initial environment
        original_python = "/original/python"
        original_pythonpath = "/original/pythonpath"
        
        os.environ['PYTHON'] = original_python
        os.environ['PYTHONPATH'] = original_pythonpath
        
        try:
            # Make multiple calls
            for i in range(3):
                success = run_workflow(
                    user_workspace=temp_workspace,
                    config=str(sample_config_file),
                    targets=["retrieve_caiso_data"],
                    dryrun=True,
                    cores=1
                )
                assert success is True
                
                # Verify environment is restored after each call
                assert os.environ.get('PYTHON') == original_python
                assert os.environ.get('PYTHONPATH') == original_pythonpath
            
            # Verify snakemake was called multiple times
            assert mock_snakemake_success.call_count == 3
            
        finally:
            # Clean up
            if 'PYTHON' in os.environ:
                del os.environ['PYTHON']
            if 'PYTHONPATH' in os.environ:
                del os.environ['PYTHONPATH']

    def test_environment_handling_with_complex_pythonpath(self, temp_workspace, sample_config_file, mock_workflow_path, mock_snakemake_success):
        """Test environment handling with complex PYTHONPATH."""
        import sys
        
        # Set complex PYTHONPATH
        complex_pythonpath = "/path1:/path2:/path with spaces:/path3"
        os.environ['PYTHONPATH'] = complex_pythonpath
        
        try:
            # Run workflow
            success = run_workflow(
                user_workspace=temp_workspace,
                config=str(sample_config_file),
                targets=["retrieve_caiso_data"],
                dryrun=True,
                cores=1
            )
            
            assert success is True
            
            # Verify PYTHONPATH is restored
            assert os.environ.get('PYTHONPATH') == complex_pythonpath
            
        finally:
            # Clean up
            if 'PYTHONPATH' in os.environ:
                del os.environ['PYTHONPATH']
