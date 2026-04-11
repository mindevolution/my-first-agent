"""A simple module that provides a greeting function."""

def say_hello(name: str = "World") -> str:
    """Return a personalized greeting string.

    Args:
        name: The name to greet. Defaults to "World".

    Returns:
        A formatted greeting string.
    """
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(say_hello())
