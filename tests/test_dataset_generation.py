import csv
import os
import random
import tempfile
import unittest

from data_generation import DataSetGenerator
from number_words import number_to_words


class TestDatasetGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        generator = DataSetGenerator(output_dir=cls.temp_dir.name)
        generator.generate_all()
        cls.parsers = {
            "int": lambda value: int(value),
        }
        cls.rows = {}
        cls.values = {}
        for format_name in cls.parsers:
            cls.rows[format_name] = {}
            cls.values[format_name] = {}
            for split in ("train", "test", "eval"):
                path = os.path.join(cls.temp_dir.name, f"{format_name}-{split}.csv")
                with open(path, newline="", encoding="utf-8") as handle:
                    rows = [row for row in csv.reader(handle) if row]
                cls.rows[format_name][split] = rows
                parser = cls.parsers[format_name]
                cls.values[format_name][split] = {parser(row[0]) for row in rows}

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def _sample_rows(self, rows, count, rng):
        if len(rows) <= count:
            return rows
        return rng.sample(rows, count)

    def test_label_matches_spelled_length(self):
        rng = random.Random(1729)
        for format_name, parser in self.parsers.items():
            for split in ("train", "test", "eval"):
                rows = self._sample_rows(self.rows[format_name][split], 100, rng)
                for value_str, label_str in rows:
                    value = parser(value_str)
                    expected = len(number_to_words(value))
                    self.assertEqual(expected, int(label_str))

    def test_train_values_in_range(self):
        rng = random.Random(2024)
        for format_name, parser in self.parsers.items():
            rows = self._sample_rows(self.rows[format_name]["train"], 100, rng)
            for value_str, _ in rows:
                value = parser(value_str)
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 999999999)

    def test_train_test_disjoint(self):
        for format_name in self.parsers:
            train_values = self.values[format_name]["train"]
            test_values = self.values[format_name]["test"]
            self.assertFalse(train_values.intersection(test_values))

    def test_train_contains_0_to_1000(self):
        for format_name in self.parsers:
            train_values = self.values[format_name]["train"]
            for value in range(0, 1001):
                self.assertIn(value, train_values)

    def test_eval_disjoint_from_train_test(self):
        for format_name in self.parsers:
            eval_values = self.values[format_name]["eval"]
            train_values = self.values[format_name]["train"]
            test_values = self.values[format_name]["test"]
            self.assertFalse(eval_values.intersection(train_values))
            self.assertFalse(eval_values.intersection(test_values))


if __name__ == "__main__":
    unittest.main()
