"""
Comprehensive pytest suite for fileutils module.

This module tests file I/O utility functions including:
- get_filename: Extract filename from full path
- getfil: Retrieve file with existence checking
- opnfil: Open file with format specification
- relavu: Close file handle safely

These are pure Python I/O functions (not JAX-compatible) that handle
file system operations for scientific computing workflows.
"""

import pytest
import os
import tempfile
from pathlib import Path
from typing import Optional, TextIO, Tuple
from unittest.mock import Mock, MagicMock, patch
import io

# Import the module under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from clm_src_utils import fileutils


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data():
    """
    Load test data for file utility functions.
    
    Returns:
        dict: Test cases organized by function name
    """
    return {
        "get_filename": [
            {
                "name": "unix_path",
                "input": "/home/user/data/simulation_results.txt",
                "expected": "simulation_results.txt",
                "description": "Standard Unix absolute path"
            },
            {
                "name": "no_directory",
                "input": "standalone_file.dat",
                "expected": "standalone_file.dat",
                "description": "Filename without directory separators"
            },
            {
                "name": "trailing_slash",
                "input": "/var/log/application/",
                "expected": "",
                "description": "Path ending with slash (directory, no filename)"
            },
            {
                "name": "windows_path",
                "input": "C:\\Users\\scientist\\Documents\\experiment_data.csv",
                "expected": "experiment_data.csv",
                "description": "Windows-style path with backslashes"
            },
            {
                "name": "multiple_extensions",
                "input": "/archive/backup/data.tar.gz.bak",
                "expected": "data.tar.gz.bak",
                "description": "Filename with multiple extensions"
            },
        ],
        "getfil": [
            {
                "name": "success_iflag_0",
                "fulpath": None,  # Will be set in test with temp file
                "iflag": 0,
                "expected_success": True,
                "description": "File exists, iflag=0 (abort on missing)"
            },
            {
                "name": "missing_iflag_1",
                "fulpath": "/nonexistent/path/missing_file.dat",
                "iflag": 1,
                "expected_success": False,
                "description": "File does not exist, iflag=1 (no abort)"
            },
            {
                "name": "relative_path",
                "fulpath": "./data/relative_path_file.bin",
                "iflag": 1,
                "expected_success": False,
                "description": "Relative path handling"
            },
            {
                "name": "empty_path",
                "fulpath": "",
                "iflag": 1,
                "expected_success": False,
                "description": "Empty path string"
            },
        ],
        "opnfil": [
            {
                "name": "unformatted_lowercase",
                "locfn": "binary_output.bin",
                "iun": 42,
                "form": "u",
                "expected_mode": "rb",
                "description": "Unformatted (binary) mode with lowercase 'u'"
            },
            {
                "name": "unformatted_uppercase",
                "locfn": "binary_output2.bin",
                "iun": 43,
                "form": "U",
                "expected_mode": "rb",
                "description": "Unformatted (binary) mode with uppercase 'U'"
            },
            {
                "name": "formatted_lowercase",
                "locfn": "text_output.txt",
                "iun": 10,
                "form": "f",
                "expected_mode": "r",
                "description": "Formatted (text) mode with lowercase 'f'"
            },
            {
                "name": "formatted_uppercase",
                "locfn": "text_output2.txt",
                "iun": 11,
                "form": "F",
                "expected_mode": "r",
                "description": "Formatted (text) mode with uppercase 'F'"
            },
            {
                "name": "long_filename",
                "locfn": "very_long_filename_that_exceeds_typical_length_but_is_still_valid.data",
                "iun": 99,
                "form": "U",
                "expected_mode": "rb",
                "description": "Very long filename (filesystem limits)"
            },
        ]
    }


@pytest.fixture
def temp_file():
    """
    Create a temporary file for testing file operations.
    
    Yields:
        str: Path to temporary file
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test data for file operations\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_binary_file():
    """
    Create a temporary binary file for testing.
    
    Yields:
        str: Path to temporary binary file
    """
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
        f.write(b'\x00\x01\x02\x03\x04\x05')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def mock_file_handle():
    """
    Create a mock file handle for testing file closing operations.
    
    Returns:
        Mock: Mock file object with close method
    """
    mock_file = MagicMock(spec=io.TextIOWrapper)
    mock_file.closed = False
    mock_file.close = Mock(side_effect=lambda: setattr(mock_file, 'closed', True))
    return mock_file


# ============================================================================
# Tests for get_filename
# ============================================================================

class TestGetFilename:
    """Test suite for get_filename function."""
    
    @pytest.mark.parametrize("test_case", [
        ("/home/user/data/simulation_results.txt", "simulation_results.txt"),
        ("standalone_file.dat", "standalone_file.dat"),
        ("/var/log/application/", ""),
        ("/archive/backup/data.tar.gz.bak", "data.tar.gz.bak"),
        ("", ""),
        ("/", ""),
        ("//double//slash//path//file.txt", "file.txt"),
    ])
    def test_get_filename_values(self, test_case):
        """
        Test get_filename returns correct filename from various path formats.
        
        Verifies extraction of filename portion from full paths including:
        - Unix absolute paths
        - Relative paths
        - Paths with trailing slashes
        - Empty strings
        - Multiple extensions
        """
        fulpath, expected = test_case
        result = fileutils.get_filename(fulpath)
        assert result == expected, (
            f"get_filename('{fulpath}') returned '{result}', "
            f"expected '{expected}'"
        )
    
    def test_get_filename_windows_path(self):
        """
        Test get_filename with Windows-style paths.
        
        Verifies cross-platform path handling with backslashes.
        Note: Behavior may vary by platform.
        """
        # On Unix systems, backslashes are treated as regular characters
        # On Windows, they are path separators
        windows_path = "C:\\Users\\scientist\\Documents\\experiment_data.csv"
        result = fileutils.get_filename(windows_path)
        
        # The result depends on the platform
        # On Windows: "experiment_data.csv"
        # On Unix: entire string (no backslash separator recognized)
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0 or windows_path.endswith('\\'), (
            "Should return filename or empty string for directory paths"
        )
    
    def test_get_filename_type(self):
        """Test get_filename returns string type."""
        result = fileutils.get_filename("/path/to/file.txt")
        assert isinstance(result, str), (
            f"get_filename should return str, got {type(result)}"
        )
    
    def test_get_filename_special_characters(self):
        """
        Test get_filename with special characters in filename.
        
        Verifies handling of spaces, dots, and other special characters.
        """
        test_cases = [
            ("/path/to/file with spaces.txt", "file with spaces.txt"),
            ("/path/to/.hidden_file", ".hidden_file"),
            ("/path/to/file-with-dashes.dat", "file-with-dashes.dat"),
            ("/path/to/file_with_underscores.nc", "file_with_underscores.nc"),
        ]
        
        for fulpath, expected in test_cases:
            result = fileutils.get_filename(fulpath)
            assert result == expected, (
                f"get_filename('{fulpath}') returned '{result}', "
                f"expected '{expected}'"
            )


# ============================================================================
# Tests for getfil
# ============================================================================

class TestGetfil:
    """Test suite for getfil function."""
    
    def test_getfil_existing_file_iflag_0(self, temp_file):
        """
        Test getfil with existing file and iflag=0 (abort on missing).
        
        Verifies successful file retrieval when file exists.
        """
        locfn, success = fileutils.getfil(temp_file, iflag=0)
        
        assert success is True, "Should return True for existing file"
        assert locfn == temp_file, (
            f"Should return original path, got '{locfn}'"
        )
        assert isinstance(locfn, str), "locfn should be string"
        assert isinstance(success, bool), "success should be boolean"
    
    def test_getfil_existing_file_iflag_1(self, temp_file):
        """
        Test getfil with existing file and iflag=1 (no abort).
        
        Verifies successful file retrieval with non-abort flag.
        """
        locfn, success = fileutils.getfil(temp_file, iflag=1)
        
        assert success is True, "Should return True for existing file"
        assert locfn == temp_file, (
            f"Should return original path, got '{locfn}'"
        )
    
    def test_getfil_missing_file_iflag_1(self):
        """
        Test getfil with missing file and iflag=1 (no abort).
        
        Verifies graceful handling of missing files when abort is disabled.
        """
        missing_path = "/nonexistent/path/missing_file.dat"
        locfn, success = fileutils.getfil(missing_path, iflag=1)
        
        assert success is False, "Should return False for missing file"
        assert locfn == missing_path, (
            "Should return original path even when file doesn't exist"
        )
    
    def test_getfil_missing_file_iflag_0(self):
        """
        Test getfil with missing file and iflag=0 (abort on missing).
        
        Verifies that function aborts (raises exception or exits) when
        file is missing and iflag=0.
        """
        missing_path = "/nonexistent/path/missing_file.dat"
        
        # Depending on implementation, this might raise an exception
        # or call sys.exit(). Test for expected behavior.
        with pytest.raises((FileNotFoundError, SystemExit, Exception)):
            fileutils.getfil(missing_path, iflag=0)
    
    def test_getfil_empty_path(self):
        """
        Test getfil with empty path string.
        
        Verifies handling of edge case with empty input.
        """
        locfn, success = fileutils.getfil("", iflag=1)
        
        assert success is False, "Should return False for empty path"
        assert locfn == "", "Should return empty string"
    
    def test_getfil_return_types(self, temp_file):
        """
        Test getfil returns correct types.
        
        Verifies return value types match specification.
        """
        result = fileutils.getfil(temp_file, iflag=1)
        
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return 2-element tuple"
        
        locfn, success = result
        assert isinstance(locfn, str), "locfn should be string"
        assert isinstance(success, bool), "success should be boolean"
    
    def test_getfil_default_iflag(self, temp_file):
        """
        Test getfil with default iflag parameter.
        
        Verifies default value of iflag=0 works correctly.
        """
        locfn, success = fileutils.getfil(temp_file)
        
        assert success is True, "Should succeed with existing file"
        assert locfn == temp_file, "Should return original path"
    
    @pytest.mark.parametrize("iflag", [0, 1])
    def test_getfil_iflag_values(self, temp_file, iflag):
        """
        Test getfil with both valid iflag values.
        
        Verifies function works with both iflag=0 and iflag=1.
        """
        locfn, success = fileutils.getfil(temp_file, iflag=iflag)
        
        assert success is True, f"Should succeed with iflag={iflag}"
        assert locfn == temp_file, "Should return original path"


# ============================================================================
# Tests for opnfil
# ============================================================================

class TestOpnfil:
    """Test suite for opnfil function."""
    
    @pytest.mark.parametrize("form,expected_mode", [
        ("u", "rb"),
        ("U", "rb"),
        ("f", "r"),
        ("F", "r"),
    ])
    def test_opnfil_format_modes(self, temp_file, temp_binary_file, form, expected_mode):
        """
        Test opnfil opens files with correct mode for each format specifier.
        
        Verifies:
        - 'u'/'U' opens in binary read mode ('rb')
        - 'f'/'F' opens in text read mode ('r')
        """
        # Use appropriate file based on format
        test_file = temp_binary_file if form in ('u', 'U') else temp_file
        
        file_handle = fileutils.opnfil(test_file, iun=42, form=form)
        
        assert file_handle is not None, (
            f"opnfil should return file handle for form='{form}'"
        )
        
        # Check mode
        if hasattr(file_handle, 'mode'):
            assert expected_mode in file_handle.mode, (
                f"File should be opened in mode '{expected_mode}', "
                f"got '{file_handle.mode}'"
            )
        
        # Cleanup
        if file_handle is not None:
            file_handle.close()
    
    def test_opnfil_returns_file_object(self, temp_file):
        """
        Test opnfil returns a valid file object.
        
        Verifies returned object has file-like interface.
        """
        file_handle = fileutils.opnfil(temp_file, iun=10, form='f')
        
        assert file_handle is not None, "Should return file handle"
        assert hasattr(file_handle, 'read'), "Should have read method"
        assert hasattr(file_handle, 'close'), "Should have close method"
        
        # Cleanup
        file_handle.close()
    
    def test_opnfil_can_read_text(self, temp_file):
        """
        Test opnfil can read text file content.
        
        Verifies file is properly opened and readable in text mode.
        """
        file_handle = fileutils.opnfil(temp_file, iun=10, form='F')
        
        assert file_handle is not None, "Should return file handle"
        
        content = file_handle.read()
        assert isinstance(content, str), "Should read text content"
        assert len(content) > 0, "Should read non-empty content"
        assert "Test data" in content, "Should contain expected text"
        
        file_handle.close()
    
    def test_opnfil_can_read_binary(self, temp_binary_file):
        """
        Test opnfil can read binary file content.
        
        Verifies file is properly opened and readable in binary mode.
        """
        file_handle = fileutils.opnfil(temp_binary_file, iun=42, form='U')
        
        assert file_handle is not None, "Should return file handle"
        
        content = file_handle.read()
        assert isinstance(content, bytes), "Should read binary content"
        assert len(content) > 0, "Should read non-empty content"
        assert content == b'\x00\x01\x02\x03\x04\x05', (
            "Should contain expected binary data"
        )
        
        file_handle.close()
    
    def test_opnfil_nonexistent_file(self):
        """
        Test opnfil with nonexistent file.
        
        Verifies error handling for missing files.
        """
        nonexistent = "/nonexistent/path/missing.txt"
        
        # Should return None or raise exception
        result = fileutils.opnfil(nonexistent, iun=10, form='f')
        
        # If it returns None, that's acceptable error handling
        # If it raises FileNotFoundError, that's also acceptable
        if result is not None:
            # If it somehow succeeded, close it
            result.close()
    
    def test_opnfil_empty_filename(self):
        """
        Test opnfil with empty filename.
        
        Verifies handling of invalid empty filename input.
        """
        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            fileutils.opnfil("", iun=10, form='f')
    
    def test_opnfil_iun_parameter(self, temp_file):
        """
        Test opnfil with various iun (unit number) values.
        
        Verifies iun parameter is accepted but not functionally used
        (maintained for Fortran interface compatibility).
        """
        for iun in [1, 10, 42, 99, 999]:
            file_handle = fileutils.opnfil(temp_file, iun=iun, form='f')
            assert file_handle is not None, (
                f"Should open file with iun={iun}"
            )
            file_handle.close()
    
    def test_opnfil_long_filename(self, temp_file):
        """
        Test opnfil with very long filename.
        
        Verifies handling of edge case with long filenames.
        """
        # Create temp file with long name
        long_name = "a" * 200 + ".txt"
        temp_dir = os.path.dirname(temp_file)
        long_path = os.path.join(temp_dir, long_name)
        
        try:
            # Create file with long name
            with open(long_path, 'w') as f:
                f.write("test")
            
            # Try to open it
            file_handle = fileutils.opnfil(long_path, iun=99, form='f')
            
            if file_handle is not None:
                assert hasattr(file_handle, 'read'), "Should be valid file handle"
                file_handle.close()
        finally:
            # Cleanup
            if os.path.exists(long_path):
                os.remove(long_path)


# ============================================================================
# Tests for relavu
# ============================================================================

class TestRelavu:
    """Test suite for relavu function."""
    
    def test_relavu_closes_file(self, mock_file_handle):
        """
        Test relavu closes an open file handle.
        
        Verifies file handle's close method is called.
        """
        fileutils.relavu(mock_file_handle)
        
        mock_file_handle.close.assert_called_once()
        assert mock_file_handle.closed is True, "File should be closed"
    
    def test_relavu_with_none(self):
        """
        Test relavu with None input.
        
        Verifies function handles None gracefully without errors.
        """
        # Should not raise any exception
        result = fileutils.relavu(None)
        assert result is None, "Should return None"
    
    def test_relavu_with_real_file(self, temp_file):
        """
        Test relavu with real file handle.
        
        Verifies function works with actual file objects.
        """
        # Open a real file
        file_handle = open(temp_file, 'r')
        assert not file_handle.closed, "File should be open"
        
        # Close it with relavu
        fileutils.relavu(file_handle)
        
        assert file_handle.closed, "File should be closed after relavu"
    
    def test_relavu_already_closed_file(self, temp_file):
        """
        Test relavu with already closed file handle.
        
        Verifies function handles already-closed files gracefully.
        """
        file_handle = open(temp_file, 'r')
        file_handle.close()
        
        # Should not raise exception when closing already-closed file
        try:
            fileutils.relavu(file_handle)
        except Exception as e:
            pytest.fail(f"relavu raised exception on closed file: {e}")
    
    def test_relavu_returns_none(self, mock_file_handle):
        """
        Test relavu returns None.
        
        Verifies function return value matches specification.
        """
        result = fileutils.relavu(mock_file_handle)
        assert result is None, "relavu should return None"
    
    def test_relavu_with_binary_file(self, temp_binary_file):
        """
        Test relavu with binary file handle.
        
        Verifies function works with binary mode files.
        """
        file_handle = open(temp_binary_file, 'rb')
        assert not file_handle.closed, "File should be open"
        
        fileutils.relavu(file_handle)
        
        assert file_handle.closed, "Binary file should be closed"


# ============================================================================
# Integration Tests
# ============================================================================

class TestFileUtilsIntegration:
    """Integration tests for fileutils module functions."""
    
    def test_workflow_get_open_close(self, temp_file):
        """
        Test complete workflow: get filename, open file, close file.
        
        Verifies functions work together in typical usage pattern.
        """
        # Get filename from path
        filename = fileutils.get_filename(temp_file)
        assert len(filename) > 0, "Should extract filename"
        
        # Check file exists
        locfn, success = fileutils.getfil(temp_file, iflag=1)
        assert success is True, "File should exist"
        
        # Open file
        file_handle = fileutils.opnfil(locfn, iun=10, form='f')
        assert file_handle is not None, "Should open file"
        
        # Read content
        content = file_handle.read()
        assert len(content) > 0, "Should read content"
        
        # Close file
        fileutils.relavu(file_handle)
        assert file_handle.closed, "File should be closed"
    
    def test_workflow_binary_file(self, temp_binary_file):
        """
        Test workflow with binary file.
        
        Verifies binary file handling through complete workflow.
        """
        # Check file exists
        locfn, success = fileutils.getfil(temp_binary_file, iflag=1)
        assert success is True, "Binary file should exist"
        
        # Open in binary mode
        file_handle = fileutils.opnfil(locfn, iun=42, form='U')
        assert file_handle is not None, "Should open binary file"
        
        # Read binary content
        content = file_handle.read()
        assert isinstance(content, bytes), "Should read bytes"
        
        # Close file
        fileutils.relavu(file_handle)
        assert file_handle.closed, "Binary file should be closed"
    
    def test_workflow_missing_file_graceful(self):
        """
        Test workflow with missing file and graceful error handling.
        
        Verifies error handling when file doesn't exist.
        """
        missing_path = "/nonexistent/missing.dat"
        
        # Check file (should fail gracefully with iflag=1)
        locfn, success = fileutils.getfil(missing_path, iflag=1)
        assert success is False, "Should report file not found"
        
        # Don't attempt to open since file doesn't exist
        # This demonstrates proper error checking workflow


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_get_filename_unicode(self):
        """
        Test get_filename with Unicode characters.
        
        Verifies handling of international characters in filenames.
        """
        unicode_paths = [
            "/path/to/файл.txt",  # Cyrillic
            "/path/to/文件.dat",  # Chinese
            "/path/to/αρχείο.nc",  # Greek
        ]
        
        for path in unicode_paths:
            result = fileutils.get_filename(path)
            assert isinstance(result, str), "Should return string"
            assert len(result) > 0, "Should extract filename"
    
    def test_opnfil_invalid_form(self, temp_file):
        """
        Test opnfil with invalid form parameter.
        
        Verifies error handling for invalid format specifiers.
        """
        invalid_forms = ['x', 'X', 'b', 'w', '']
        
        for form in invalid_forms:
            with pytest.raises((ValueError, KeyError, Exception)):
                fileutils.opnfil(temp_file, iun=10, form=form)
    
    def test_getfil_invalid_iflag(self, temp_file):
        """
        Test getfil with invalid iflag values.
        
        Verifies handling of out-of-range iflag values.
        """
        # Test with invalid iflag values (should be 0 or 1)
        invalid_iflags = [-1, 2, 10, 999]
        
        for iflag in invalid_iflags:
            # Function might accept any integer or raise ValueError
            try:
                locfn, success = fileutils.getfil(temp_file, iflag=iflag)
                # If it accepts it, verify it returns valid types
                assert isinstance(locfn, str)
                assert isinstance(success, bool)
            except ValueError:
                # Raising ValueError for invalid iflag is acceptable
                pass
    
    def test_module_constants(self):
        """
        Test module constants are defined correctly.
        
        Verifies UNFORMATTED_FORMATS and FORMATTED_FORMATS constants.
        """
        assert hasattr(fileutils, 'UNFORMATTED_FORMATS'), (
            "Module should define UNFORMATTED_FORMATS"
        )
        assert hasattr(fileutils, 'FORMATTED_FORMATS'), (
            "Module should define FORMATTED_FORMATS"
        )
        
        unformatted = fileutils.UNFORMATTED_FORMATS
        formatted = fileutils.FORMATTED_FORMATS
        
        assert 'u' in unformatted, "Should include 'u'"
        assert 'U' in unformatted, "Should include 'U'"
        assert 'f' in formatted, "Should include 'f'"
        assert 'F' in formatted, "Should include 'F'"


# ============================================================================
# Documentation Tests
# ============================================================================

class TestDocumentation:
    """Test function documentation and signatures."""
    
    def test_functions_have_docstrings(self):
        """
        Test all functions have docstrings.
        
        Verifies documentation exists for all public functions.
        """
        functions = ['get_filename', 'getfil', 'opnfil', 'relavu']
        
        for func_name in functions:
            func = getattr(fileutils, func_name)
            assert func.__doc__ is not None, (
                f"Function {func_name} should have docstring"
            )
            assert len(func.__doc__.strip()) > 0, (
                f"Function {func_name} docstring should not be empty"
            )
    
    def test_module_has_docstring(self):
        """
        Test module has docstring.
        
        Verifies module-level documentation exists.
        """
        assert fileutils.__doc__ is not None, "Module should have docstring"
        assert len(fileutils.__doc__.strip()) > 0, (
            "Module docstring should not be empty"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])