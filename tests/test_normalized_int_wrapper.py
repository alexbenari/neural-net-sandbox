import unittest

import torch

from data_generation import NormalizedIntWrapper


class _ValueDataset:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        value = torch.tensor([float(self.values[index])], dtype=torch.float32)
        target = torch.tensor([0.0], dtype=torch.float32)
        return value, target


class TestNormalizedIntWrapper(unittest.TestCase):
    def test_normalizes_by_default_divisor(self):
        dataset = _ValueDataset([0, 500_000_000, 1_000_000_000])
        wrapper = NormalizedIntWrapper(dataset)
        value, _ = wrapper[1]
        self.assertAlmostEqual(value.item(), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
