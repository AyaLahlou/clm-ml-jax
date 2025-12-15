"""
CLM variable control module

This module contains run control variables used throughout CLM.
Translated from Fortran CLM code to Python JAX.
"""

import sys
import logging
from typing import Union, TextIO, Optional
from io import TextIOWrapper

# Import dependencies
from ..cime_src_share_util.shr_kind_mod import r8


class CLMLogHandler:
    """
    Handler for CLM logging operations
    
    This class provides a flexible logging interface that can work with
    file handles, logging objects, or standard output.
    """
    
    def __init__(self, log_unit: Union[int, TextIO, logging.Logger] = 6):
        """
        Initialize the log handler
        
        Args:
            log_unit: Log unit - can be an integer (6=stdout), file handle, or logger
        """
        self._log_unit = log_unit
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup the logging configuration"""
        if isinstance(self._log_unit, int):
            if self._log_unit == 6:
                self._output = sys.stdout
            elif self._log_unit == 0:
                self._output = sys.stderr
            else:
                # For other integer values, create a file
                try:
                    self._output = open(f"clm_log_{self._log_unit}.txt", "w")
                except:
                    self._output = sys.stdout
        elif isinstance(self._log_unit, (TextIO, TextIOWrapper)):
            self._output = self._log_unit
        elif isinstance(self._log_unit, logging.Logger):
            self._output = self._log_unit
        else:
            self._output = sys.stdout
    
    def write(self, message: str) -> None:
        """
        Write a message to the log
        
        Args:
            message: Message to write
        """
        if isinstance(self._output, logging.Logger):
            self._output.info(message.rstrip('\n'))
        else:
            self._output.write(message)
            if hasattr(self._output, 'flush'):
                self._output.flush()
    
    def error(self, message: str) -> None:
        """
        Write an error message to the log
        
        Args:
            message: Error message to write
        """
        if isinstance(self._output, logging.Logger):
            self._output.error(message)
        else:
            error_msg = f"ERROR: {message}\n"
            self._output.write(error_msg)
            if hasattr(self._output, 'flush'):
                self._output.flush()
    
    def warning(self, message: str) -> None:
        """
        Write a warning message to the log
        
        Args:
            message: Warning message to write
        """
        if isinstance(self._output, logging.Logger):
            self._output.warning(message)
        else:
            warning_msg = f"WARNING: {message}\n"
            self._output.write(warning_msg)
            if hasattr(self._output, 'flush'):
                self._output.flush()
    
    def close(self) -> None:
        """Close the log handler if it's a file"""
        if hasattr(self._output, 'close') and self._output not in (sys.stdout, sys.stderr):
            self._output.close()
    
    @property 
    def unit_number(self) -> int:
        """Get the original unit number"""
        if isinstance(self._log_unit, int):
            return self._log_unit
        return 6  # Default to stdout
    
    @property
    def output_stream(self) -> Union[TextIO, logging.Logger]:
        """Get the actual output stream"""
        return self._output


# Global log unit variable (equivalent to Fortran module variable)
# Default is 6 which corresponds to stdout in Fortran
_default_log_handler = CLMLogHandler(6)
iulog = _default_log_handler


def set_log_unit(unit: Union[int, str, TextIO, logging.Logger]) -> None:
    """
    Set the global log unit
    
    Args:
        unit: New log unit - can be integer, filename, file handle, or logger
    """
    global iulog, _default_log_handler
    
    if isinstance(unit, str):
        # If it's a string, treat as filename
        try:
            file_handle = open(unit, "w")
            _default_log_handler = CLMLogHandler(file_handle)
        except:
            # Fallback to stdout if file can't be opened
            _default_log_handler = CLMLogHandler(6)
    else:
        _default_log_handler = CLMLogHandler(unit)
    
    iulog = _default_log_handler


def get_log_unit() -> CLMLogHandler:
    """
    Get the current log unit handler
    
    Returns:
        Current log handler
    """
    return iulog


def log_message(message: str, level: str = "info") -> None:
    """
    Log a message using the global log handler
    
    Args:
        message: Message to log
        level: Log level ("info", "error", "warning")
    """
    if level.lower() == "error":
        iulog.error(message)
    elif level.lower() == "warning":
        iulog.warning(message)
    else:
        iulog.write(f"{message}\n")


def close_log() -> None:
    """Close the current log handler"""
    iulog.close()


# Context manager for temporary log redirection
class TemporaryLogRedirect:
    """
    Context manager for temporarily redirecting log output
    
    Example:
        with TemporaryLogRedirect("debug.log"):
            # All CLM log output goes to debug.log
            log_message("This goes to debug.log")
        # Log output restored to previous destination
    """
    
    def __init__(self, new_unit: Union[int, str, TextIO, logging.Logger]):
        self.new_unit = new_unit
        self.old_handler = None
    
    def __enter__(self):
        global iulog
        self.old_handler = iulog
        set_log_unit(self.new_unit)
        return iulog
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global iulog
        current_handler = iulog
        iulog = self.old_handler
        if current_handler != self.old_handler:
            current_handler.close()


# Configuration class for run control variables
class CLMRunControl:
    """
    Class to hold CLM run control variables and configuration
    
    This can be extended to include additional control variables
    as they are encountered in other CLM modules.
    """
    
    def __init__(self):
        self.iulog_number = 6  # Default log unit number
        self.log_handler = iulog
        
        # Placeholder for additional run control variables that might
        # be added from other CLM modules
        self.debug_level = 0
        self.verbose = False
        self.model_name = "CLM"
    
    def set_log_unit(self, unit: Union[int, str, TextIO, logging.Logger]) -> None:
        """Set log unit for this control instance"""
        set_log_unit(unit)
        self.log_handler = get_log_unit()
        if isinstance(unit, int):
            self.iulog_number = unit
    
    def get_config_dict(self) -> dict:
        """Get configuration as dictionary"""
        return {
            'iulog_number': self.iulog_number,
            'debug_level': self.debug_level,
            'verbose': self.verbose,
            'model_name': self.model_name
        }
    
    def load_config_dict(self, config: dict) -> None:
        """Load configuration from dictionary"""
        self.debug_level = config.get('debug_level', 0)
        self.verbose = config.get('verbose', False)
        self.model_name = config.get('model_name', "CLM")
        
        if 'iulog_number' in config:
            self.set_log_unit(config['iulog_number'])


# Global run control instance
clm_run_control = CLMRunControl()


# Public interface
__all__ = [
    'iulog', 'CLMLogHandler', 'CLMRunControl', 'clm_run_control',
    'set_log_unit', 'get_log_unit', 'log_message', 'close_log',
    'TemporaryLogRedirect'
]