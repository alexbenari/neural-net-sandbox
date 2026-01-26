import csv
import os
import random

import torch
from torch.utils.data import Dataset

from number_words import number_to_words


class DataSetGenerator:
    def __init__(
        self,
        output_dir="data",
        train_size=100_000,
        test_size=10_000,
        eval_size=10_000,
        min_value=0,
        max_value=999_999_999,
        digits_width=9,
        binary_width=30,
        seed=42,
    ):
        self.output_dir = output_dir
        self.train_size = train_size
        self.test_size = test_size
        self.eval_size = eval_size
        self.min_value = min_value
        self.max_value = max_value
        self.digits_width = digits_width
        self.binary_width = binary_width
        self.rng = random.Random(seed)

    def _write_csv(self, path, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)

    def _sample_train_values(self):
        values = set(range(0, 1001))
        while len(values) < self.train_size:
            value = self.rng.randint(self.min_value, self.max_value)
            values.add(value)
        return values

    def _sample_unique(self, size, low, high, excluded):
        values = set()
        while len(values) < size:
            value = self.rng.randint(low, high)
            if value not in excluded:
                values.add(value)
        return values

    def _generate_values(self):
        eval_seed_values = set(range(0, 1001))
        train_values = self._sample_train_values()
        test_values = self._sample_unique(
            self.test_size,
            self.min_value,
            self.max_value,
            train_values,
        )
        eval_values = set()
        excluded_eval = set(train_values)
        excluded_eval.update(test_values)
        eval_random_values = self._sample_unique(
            self.eval_size,
            self.min_value,
            self.max_value,
            excluded_eval,
        )
        eval_values.update(eval_random_values)
        return {
            "train": sorted(train_values),
            "test": sorted(test_values),
            "eval": sorted(eval_values),
        }

    def _label_for_value(self, value):
        return len(number_to_words(value))

    def _write_split(self, format_name, formatter, values_by_split):
        for split_name, values in values_by_split.items():
            rows = [
                (formatter(value), str(self._label_for_value(value)))
                for value in values
            ]
            filename = f"{format_name}-{split_name}.csv"
            path = os.path.join(self.output_dir, filename)
            self._write_csv(path, rows)
            print(f"Wrote {filename} with {len(rows)} rows.")

    def generate_int_format(self, values_by_split):
        self._write_split("int", lambda value: str(value), values_by_split)

    def generate_all(self):
        values_by_split = self._generate_values()
        self.generate_int_format(values_by_split)


class IntDataset(Dataset):
    def __init__(self, csv_path):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if row:
                    self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        value_str, target_str = self.rows[index]
        value = torch.tensor([float(value_str)], dtype=torch.float32)
        target = torch.tensor([float(target_str)], dtype=torch.float32)
        return value, target


class DigitOneHotWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        digits, target = self.dataset[index]
        vector = torch.zeros(digits.numel() * 10, dtype=digits.dtype)
        for idx, value in enumerate(digits):
            vector[idx * 10 + int(value.item())] = 1.0
        return vector, target


class NormalizedIntWrapper(Dataset):
    def __init__(self, dataset, divisor=1_000_000_000):
        self.dataset = dataset
        self.divisor = divisor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        value, target = self.dataset[index]
        return value / self.divisor, target


class DigitsDatasetWrapper(Dataset):
    def __init__(self, dataset, digits_width=9):
        self.dataset = dataset
        self.digits_width = digits_width

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        value, target = self.dataset[index]
        digits_str = str(int(value.item())).zfill(self.digits_width)
        digits = torch.tensor([int(ch) for ch in digits_str], dtype=torch.float32)
        return digits, target


class BinaryDatasetWrapper(Dataset):
    def __init__(self, dataset, binary_width=30):
        self.dataset = dataset
        self.binary_width = binary_width

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        value, target = self.dataset[index]
        bits_str = format(int(value.item()), "b").zfill(self.binary_width)
        bits = torch.tensor([int(bit) for bit in bits_str], dtype=torch.float32)
        return bits, target


class ZeroedInputsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return torch.zeros_like(data), label
