"""

App entry point."""

import os
import sys


def main():
    """Main function."""
    os.execlp(sys.executable, sys.executable, "-m", "streamlit", "run", "44_👩_DUO_Demo.py")


if __name__ == "__main__":
    main()
