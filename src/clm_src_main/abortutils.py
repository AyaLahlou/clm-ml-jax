"""
Abort utilities module

This module provides utilities to abort the model for abnormal termination.
Translated from Fortran CLM code to Python JAX.
"""

import sys
import logging
from typing import Optional

# Optional import of netCDF4 - not needed for basic functionality
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        import netCDF4 as nc
    HAS_NETCDF4 = True
except (ImportError, RuntimeWarning):
    # Create a dummy nc module for when netCDF4 is not available
    class DummyNetCDF4:
        NF_NOERR = 0
    nc = DummyNetCDF4()
    HAS_NETCDF4 = False

# Import dependencies
try:
    from ..cime_src_share_util.shr_kind_mod import r8
    from .clm_varctl import DEFAULT_CLM_VARCTL
    # Get the default iulog value for compatibility
    iulog = DEFAULT_CLM_VARCTL.iulog
except ImportError:
    # Fallback for when running outside package context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cime_src_share_util.shr_kind_mod import r8
    from clm_src_main.clm_varctl import DEFAULT_CLM_VARCTL
    # Get the default iulog value for compatibility
    iulog = DEFAULT_CLM_VARCTL.iulog


# NetCDF constants (equivalent to netcdf.inc)
class NetCDFConstants:
    """NetCDF constants equivalent to Fortran include 'netcdf.inc'"""
    NF_NOERR = 0  # No error


def endrun(msg: Optional[str] = None) -> None:
    """
    Abort the model execution with an optional message
    
    This function terminates the model execution and prints an error message
    to the log unit. Equivalent to Fortran STOP statement.
    
    Args:
        msg: Optional string to be printed before termination
    """
    if msg is not None:
        # Write to log unit (equivalent to Fortran write(iulog,*))
        print(f"ENDRUN: {msg}")
        if hasattr(iulog, 'write'):
            iulog.write(f"ENDRUN: {msg}\n")
        else:
            # If iulog is a file handle or logger
            try:
                logging.error(f"ENDRUN: {msg}")
            except Exception:
                pass
    else:
        print("ENDRUN: called without a message string")
        if hasattr(iulog, 'write'):
            iulog.write("ENDRUN: called without a message string\n")
        else:
            try:
                logging.error("ENDRUN: called without a message string")
            except Exception:
                pass
    
    # Equivalent to Fortran STOP
    sys.exit(1)


def handle_err(status: int, errmsg: str) -> None:
    """
    Handle NetCDF errors by checking status and aborting if necessary
    
    This function checks a NetCDF status code and terminates execution
    if an error occurred, printing the NetCDF error message along with
    a custom error message.
    
    Args:
        status: NetCDF status code
        errmsg: Custom error message to append
    """
    if status != NetCDFConstants.NF_NOERR:
        # Get NetCDF error string (equivalent to nf_strerror)
        try:
            netcdf_error = nc.strerror(status)
        except (AttributeError, TypeError):
            netcdf_error = f"NetCDF error code: {status}"
        
        # Print error message (equivalent to Fortran print)
        error_message = f"{netcdf_error.strip()}: {errmsg}"
        print(error_message)
        
        # Log the error if possible
        try:
            logging.error(error_message)
        except Exception:
            pass
        
        # Terminate execution (equivalent to Fortran stop "Stopped")
        sys.exit(1)


def check_netcdf_status(status: int, operation: str = "NetCDF operation") -> None:
    """
    Convenience function to check NetCDF status with a descriptive operation name
    
    Args:
        status: NetCDF status code to check
        operation: Description of the operation that was attempted
    """
    handle_err(status, operation)


def assert_condition(condition: bool, msg: str) -> None:
    """
    Assert a condition and call endrun if it fails
    
    This is a convenience function that combines condition checking
    with the endrun functionality.
    
    Args:
        condition: Boolean condition to check
        msg: Error message to display if condition is False
    """
    if not condition:
        endrun(msg)


def warn_and_continue(msg: str) -> None:
    """
    Print a warning message but continue execution
    
    Args:
        msg: Warning message to display
    """
    warning_msg = f"WARNING: {msg}"
    print(warning_msg)
    
    if hasattr(iulog, 'write'):
        iulog.write(f"{warning_msg}\n")
    else:
        try:
            logging.warning(msg)
        except Exception:
            pass


# Exception classes for better error handling in Python
class CLMError(Exception):
    """Base exception class for CLM errors"""
    pass


class CLMNetCDFError(CLMError):
    """Exception for NetCDF-related errors in CLM"""
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"NetCDF Error {status}: {message}")


class CLMInitializationError(CLMError):
    """Exception for initialization errors in CLM"""
    pass


class CLMComputationError(CLMError):
    """Exception for computation errors in CLM"""
    pass


# Public interface
__all__ = [
    'endrun', 'handle_err', 'check_netcdf_status', 'assert_condition', 
    'warn_and_continue', 'NetCDFConstants', 
    'CLMError', 'CLMNetCDFError', 'CLMInitializationError', 'CLMComputationError'
]