"""
Main entry point for Elips Financial Planner.

This module provides the main entry point and demonstrates the core
foundation functionality of the financial planning system.
"""

import sys
import traceback
from pathlib import Path
from typing import Optional

from .config import get_config, ConfigurationError
from .logger import setup_logging, get_logger, ErrorHandler, progress_context


def init_system() -> tuple:
    """
    Initialize the financial planning system.
    
    Returns:
        Tuple of (config, logger) instances
        
    Raises:
        SystemExit: If initialization fails
    """
    try:
        # Load configuration
        config = get_config()
        
        # Setup logging based on configuration
        log_file = config.get_output_path("elips_planner.log")
        logger = setup_logging(
            level=config.get('logging.level', 'INFO'),
            log_file=log_file,
            max_file_size=config.get('logging.max_file_size', 10485760),
            backup_count=config.get('logging.backup_count', 5)
        )
        
        logger.info("=" * 60)
        logger.info("ELIPS FINANCIAL PLANNER - SYSTEM STARTUP")
        logger.info("=" * 60)
        logger.info(f"Version: {config.get('version', 'Unknown')}")
        logger.info(f"Configuration loaded: {config}")
        logger.info(f"Log file: {log_file}")
        
        return config, logger
        
    except ConfigurationError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"System Initialization Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def demonstrate_foundation(config, logger) -> None:
    """
    Demonstrate core foundation functionality.
    
    Args:
        config: Configuration instance
        logger: Logger instance
    """
    logger.info("Starting foundation demonstration...")
    
    # Demonstrate configuration access
    logger.info("Configuration demonstration:")
    logger.info(f"  Project: {config.get('project_name')}")
    logger.info(f"  Forecast model: {config.get('forecast.model')}")
    logger.info(f"  Forecast horizon: {config.get('forecast.horizon')} months")
    logger.info(f"  Tax rate: {config.get('finance.tax_rate') * 100:.1f}%")
    logger.info(f"  Available scenarios: {config.list_scenarios()}")
    
    # Demonstrate progress indicator
    logger.info("Progress indicator demonstration:")
    with progress_context(5, "Initializing modules") as progress:
        import time
        
        progress.update(1, "Loading configuration")
        time.sleep(0.5)
        
        progress.update(1, "Setting up logging")
        time.sleep(0.5)
        
        progress.update(1, "Validating parameters")
        time.sleep(0.5)
        
        progress.update(1, "Preparing data structures")
        time.sleep(0.5)
        
        progress.update(1, "System ready")
        time.sleep(0.5)
    
    # Demonstrate error handling
    logger.info("Error handling demonstration:")
    try:
        ErrorHandler.validate_input(
            value=15,
            name="test_parameter",
            expected_type=int,
            min_value=1,
            max_value=12
        )
    except ValueError as e:
        logger.warning(f"Validation error caught: {e}")
    
    # Demonstrate scenario access
    logger.info("Scenario configuration demonstration:")
    for scenario_name in config.list_scenarios():
        scenario = config.get_scenario(scenario_name)
        logger.info(f"  {scenario_name}: {scenario['description']}")
    
    # Demonstrate file path resolution
    logger.info("File path demonstration:")
    data_file = config.get_data_path("data/sales_history.csv")
    output_file = config.get_output_path("demo_output.txt")
    logger.info(f"  Data path: {data_file}")
    logger.info(f"  Output path: {output_file}")
    
    # Create demo output file
    Path(output_file).write_text(
        f"Elips Financial Planner Demo Output\n"
        f"Generated at: {sys.version}\n"
        f"Configuration: {config}\n"
        f"System Status: Foundation Ready\n"
    )
    
    logger.info(f"Demo output written to: {output_file}")
    logger.info("Foundation demonstration completed successfully!")


def show_system_info(config, logger) -> None:
    """
    Display system information and status.
    
    Args:
        config: Configuration instance
        logger: Logger instance
    """
    logger.info("SYSTEM INFORMATION")
    logger.info("-" * 40)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Check if required directories exist
    project_root = Path(__file__).parent.parent
    required_dirs = ['src', 'config', 'data', 'outputs', 'tests']
    
    logger.info("Directory structure:")
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        status = "OK" if dir_path.exists() else "MISSING"
        logger.info(f"  {status} {dir_name}/")
    
    # Check configuration sections
    logger.info("Configuration sections:")
    sections = ['forecast', 'finance', 'scenarios', 'risk', 'data', 'logging']
    for section in sections:
        try:
            config.get(section)
            logger.info(f"  OK {section}")
        except ConfigurationError:
            logger.warning(f"  MISSING {section}")


def main() -> None:
    """Main entry point for the financial planner."""
    print("Elips Medikal Financial Planning System")
    print("Data-Driven Financial Planning & Risk Analysis")
    print()
    
    try:
        # Initialize system
        config, logger = init_system()
        
        # Show system information
        show_system_info(config, logger)
        
        # Demonstrate foundation functionality
        demonstrate_foundation(config, logger)
        
        logger.info("=" * 60)
        logger.info("PHASE 1 FOUNDATION: SUCCESSFULLY COMPLETED")
        logger.info("Next: Implement data loading and forecasting models")
        logger.info("=" * 60)
        
        print("\nPhase 1 Foundation completed successfully!")
        print("Check the log file for detailed information")
        print("Ready for Phase 2: Data Foundation & Modeling")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()