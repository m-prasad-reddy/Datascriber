from cli.interface import CLIInterface, CLIError
import logging

def main():
    """Entry point for the Datascriber application.

    Initializes the CLI interface and starts the user interaction loop.

    Raises:
        CLIError: If CLI initialization or execution fails.
    """
    try:
        cli = CLIInterface()
        cli.run()
    except CLIError as e:
        logging.error(f"Application error: {str(e)}")
        print(f"Error: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()