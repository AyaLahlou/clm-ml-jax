"""
CLM main source code.

This module contains the core Community Land Model (CLM) modules including
main control structures, decomposition modules, and primary model components.

Note: Temporary version for testing with only abortutils imports.
"""

# Import only abortutils for now to avoid complex dependency chains
from .abortutils import (
    endrun, handle_err, check_netcdf_status, assert_condition, 
    warn_and_continue, NetCDFConstants,
    CLMError, CLMNetCDFError, CLMInitializationError, CLMComputationError
)
__all__ = [
    'endrun', 'handle_err', 'check_netcdf_status', 'assert_condition', 
    'warn_and_continue', 'NetCDFConstants',
    'CLMError', 'CLMNetCDFError', 'CLMInitializationError', 'CLMComputationError'
]