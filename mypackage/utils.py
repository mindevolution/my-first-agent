"""Utility functions for the mypackage package.

This module provides basic utility functions for common operations.
"""


def hello_world(name: str = "World") -> str:
    """Return a greeting message.

    Args:
        name: The name to greet. Defaults to "World".

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number to add.
        b: Second number to add.

    Returns:
        The sum of the two numbers.
    """
    return a + b


def is_even(number: int) -> bool:
    """Check if a number is even.

    Args:
        number: The number to check.

    Returns:
        True if the number is even, False otherwise.
    """
    return number % 2 == 0
