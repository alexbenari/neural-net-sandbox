import unittest

import torch

from data_generation import DigitOneHotWrapper


class _DigitsDataset:
    def __init__(self, digit_strings):
        self.rows = digit_strings

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        digits_str = self.rows[index]
        digits = torch.tensor([int(ch) for ch in digits_str], dtype=torch.float32)
        target = torch.tensor([0.0], dtype=torch.float32)
        return digits, target


class TestDigitOneHotEncoding(unittest.TestCase):
    def _assert_one_hot(self, digits_str):
        dataset = _DigitsDataset([digits_str])
        wrapper = DigitOneHotWrapper(dataset)
        vector, _ = wrapper[0]
        self.assertEqual(vector.numel(), 90)
        nonzero = torch.nonzero(vector, as_tuple=False).flatten().tolist()
        expected = [idx * 10 + int(ch) for idx, ch in enumerate(digits_str)]
        self.assertEqual(sorted(nonzero), expected)
        self.assertEqual(int(vector.sum().item()), 9)

    def test_all_zeros(self):
        self._assert_one_hot("000000000")

    def test_all_nines(self):
        self._assert_one_hot("999999999")

    def test_mixed_with_leading_zeros(self):
        self._assert_one_hot("000012002")


if __name__ == "__main__":
    unittest.main()
