# Neural Net Dataset Overview

## Purpose
This project trains a neural network to predict the length (in letters and spaces) of the English, spelled-out form of an integer. For example, `123` becomes "one hundred and twenty three", which has length `28`.

## Dataset Structure
Each dataset row has two columns:
1. The input value in a chosen format.
2. The target label: the count of letters and spaces in the spelled-out number.

Three formats are generated from the same underlying integer values:
- `int`: raw integer as a string.
- `digits`: zero-padded 9-digit string (MSB to LSB).
- `binary`: zero-padded 30-bit string (MSB to LSB).

Files are stored under `data/` with a format prefix:
`int-train.csv`, `int-test.csv`, `int-eval.csv` (and the same for `digits` and `binary`).

## Data Generation Rules
- **Train (100K)**: random integers in `0..600000000` with no digit restriction.
- **Test (10K)**: random integers in `0..600000000` with no digit restriction, excluding any train values.
- **Eval (10,000 total)**:
  - All integers `0..1000` (these may overlap with train, but not test).
  - Remaining values are random integers in `0..999000000`, excluding train/test and the `0..1000` block.

## How to Generate
- Default output directory: `python sandbox.py --generate-data`
- Custom output directory: `python sandbox.py --generate-data --output-dir data`

## Spelling Rules
Numbers use English words with spaces, including "and" in the British style:
- `123` -> "one hundred and twenty three"
- `1001` -> "one thousand and one"
The label is the length of this string, counting letters and spaces.
