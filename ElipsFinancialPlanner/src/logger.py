"""
Logging system for Elips Financial Planner.

This module provides centralized logging configuration with file rotation,
progress indicators, and error handling.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ProgressIndicator:
    """Simple progress indicator for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress indicator.
        
        Args:
            total: Total number of steps
            description: Description of the operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, step: int = 1, message: str = "") -> None:
        """
        Update progress.
        
        Args:
            step: Number of steps completed
            message: Optional status message
        """
        self.current += step
        percentage = (self.current / self.total) * 100
        
        elapsed = datetime.now() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f" ETA: {eta.total_seconds():.0f}s"
        else:
            eta_str = ""
        
        status = f"\r{self.description}: {percentage:.1f}% ({self.current}/{self.total}){eta_str}"
        if message:
            status += f" - {message}"
        
        print(status, end="", flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def finish(self, message: str = "Complete") -> None:
        """Finish progress indicator with final message."""
        self.current = self.total
        elapsed = datetime.now() - self.start_time
        print(f"\r{self.description}: {message} in {elapsed.total_seconds():.1f}s")


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[41m', # Red background
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        
        # Format the message
        formatted = super().format(record)
        
        # Add color to level name only
        if log_color:
            formatted = formatted.replace(
                record.levelname,
                f"{log_color}{record.levelname}{reset}"
            )
        
        return formatted


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10485760,  # 10MB
    backup_count: int = 5,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup centralized logging system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, no file logging)
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('elips_planner')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = ColoredFormatter(log_format, date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate messages
    logger.propagate = False
    
    return logger


def get_logger(name: str = 'elips_planner') -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ErrorHandler:
    """Centralized error handling utilities."""
    
    @staticmethod
    def handle_error(
        error: Exception,
        context: str = "",
        logger: Optional[logging.Logger] = None,
        reraise: bool = True
    ) -> None:
        """
        Handle errors with consistent logging and optional re-raising.
        
        Args:
            error: The exception that occurred
            context: Context description of where error occurred
            logger: Logger instance (if None, gets default logger)
            reraise: Whether to re-raise the exception
        """
        if logger is None:
            logger = get_logger()
        
        error_msg = f"Error in {context}: {type(error).__name__}: {str(error)}"
        logger.error(error_msg)
        
        if reraise:
            raise error
    
    @staticmethod
    def validate_input(
        value: any,
        name: str,
        expected_type: type = None,
        min_value: float = None,
        max_value: float = None,
        allowed_values: list = None
    ) -> None:
        """
        Validate input parameters with clear error messages.
        
        Args:
            value: Value to validate
            name: Parameter name for error messages
            expected_type: Expected type
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allowed_values: List of allowed values
            
        Raises:
            ValueError: If validation fails
        """
        if expected_type and not isinstance(value, expected_type):
            raise ValueError(f"{name} must be {expected_type.__name__}, got {type(value).__name__}")
        
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {value}")
        
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} must be <= {max_value}, got {value}")
        
        if allowed_values is not None and value not in allowed_values:
            raise ValueError(f"{name} must be one of {allowed_values}, got {value}")


# Context manager for progress tracking
class progress_context:
    """Context manager for progress tracking."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """Initialize progress context."""
        self.progress = ProgressIndicator(total, description)
    
    def __enter__(self):
        """Enter context."""
        return self.progress
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is None:
            self.progress.finish()
        else:
            self.progress.finish("Failed")
            return False  # Don't suppress exceptions