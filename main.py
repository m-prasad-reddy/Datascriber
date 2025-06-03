from config.utils import ConfigUtils
from config.logging_setup import LoggingSetup
from cli.interface import Interface

def main():
    """Main entry point for the Datascriber system."""
    try:
        # Initialize ConfigUtils and LoggingSetup once
        config_utils = ConfigUtils()
        logging_setup = LoggingSetup.get_instance(config_utils)
        logger = logging_setup.get_logger("main", "system")
        
        logger.debug("Starting Datascriber CLI")
        cli = Interface(config_utils)  # Pass config_utils to Interface
        cli.run()
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()