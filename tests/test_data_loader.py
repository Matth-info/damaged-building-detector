# tests/test_data_loader.py
import unittest
from src.data.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def test_data_loading(self):
        loader = DataLoader("data/raw")
        data = loader.load_data()
        self.assertIsNotNone(data)

if __name__ == '__main__':
    unittest.main()