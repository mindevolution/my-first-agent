"""A simple hello world module.

This module demonstrates basic Python practices including
type hints, docstrings, and the main guard pattern.
"""


def say_hello(name: str = "World") -> str:
    """Return a greeting message for the given name.

    Args:
        name: The name to greet. Defaults to "World".

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


def main() -> None:
    """Main entry point of the program.

    Prints a greeting message to the console.
    """
    print(say_hello())


if __name__ == "__main__":
    main()
