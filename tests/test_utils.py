"""Tests for the mypackage.utils module.

This test file contains unit tests for the utility functions
in the mypackage.utils module.
"""

import unittest
from mypackage.utils import hello_world, add_numbers, is_even


class TestUtils(unittest.TestCase):
    """Test cases for the utils module."""

    def test_hello_world(self) -> None:
        """Test the hello_world function."""
        self.assertEqual(hello_world(), "Hello, World!")
        self.assertEqual(hello_world("Alice"), "Hello, Alice!")

    def test_add_numbers(self) -> None:
        """Test the add_numbers function."""
        self.assertEqual(add_numbers(2.0, 3.0), 5.0)
        self.assertEqual(add_numbers(-1.5, 2.5), 1.0)

    def test_is_even(self) -> None:
        """Test the is_even function."""
        self.assertTrue(is_even(4))
        self.assertFalse(is_even(3))
        self.assertTrue(is_even(0))
        self.assertFalse(is_even(-1))


if __name__ == "__main__":
    unittest.main()
