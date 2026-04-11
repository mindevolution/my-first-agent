# mypackage/tests/test_utils.py
import unittest
from .. import utils

class TestUtils(unittest.TestCase):
    def test_add(self):
        self.assertEqual(utils.add(2, 3), 5)

    def test_multiply(self):
        self.assertEqual(utils.multiply(2, 3), 6)


if __name__ == "__main__":
    unittest.main()
