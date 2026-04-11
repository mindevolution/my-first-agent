def say_hello(name: str = "World") -> str:
    """
    Say hello to a given name.

    Args:
        name: The name to greet. Defaults to "World".

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


def main() -> None:
    """
    Main function to demonstrate the say_hello function.
    """
    print(say_hello())
    print(say_hello("Alice"))


if __name__ == "__main__":
    main()
