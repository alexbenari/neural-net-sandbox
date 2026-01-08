import unittest

from number_words import number_to_words


class TestNumberToWords(unittest.TestCase):
    def test_examples(self):
        cases = {
            0: "zero",
            10: "ten",
            99: "ninety nine",
            100: "one hundred",
            1001: "one thousand and one",
            10123: "ten thousand one hundred and twenty three",
            251334506: (
                "two hundred and fifty one million "
                "three hundred and thirty four thousand "
                "five hundred and six"
            ),
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                words = number_to_words(value)
                self.assertEqual(words, expected)
                self.assertEqual(len(words), len(expected))


if __name__ == "__main__":
    unittest.main()
