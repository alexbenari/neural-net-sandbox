# Neural Net Dataset Overview

## Purpose
This project trains a neural network to predict the length (in letters and spaces) of the English, spelled-out form of an integer. For example, `123` becomes "one hundred and twenty three", which has length `28`.

## Dataset Structure
Each dataset row has two columns:
1. The input number
2. The target label, i.e. the count of letters and spaces in the spelled-out number.

An additional goal of this project is to experiment with the effect of input representation on training. Thus, the integer values are transformed as part of the dataset creatrion into one of four representations: 
- normalized-int: the input integer value (parsed from the CSV). For training it is converted to a numeric tensor (float) and normalized (e.g., divided by 1000000000) before being passed to the model.
-digits: a fixed-length sequence of 9 decimal digits representing a non-negative integer in base 10, left-zero-padded and ordered from most significant to least significant digit; each digit is subsequently encoded as an integer in {0,…,9}.
- digit1h: a fixed-length representation of a non-negative integer in base-10, obtained by left-padding the integer to 9 digits and encoding each digit (from most significant to least significant) as a 10-dimensional one-hot vector; the final representation is the concatenation of these 9 vectors (total dimension 90).
- binary: a fixed-length representation of a non-negative integer obtained by encoding its base-2 representation as a 30-dimensional vector of bits, ordered from most significant to least significant, with left-zero-padding applied to integers whose binary representation has fewer than 30 bits.

Files are stored under `data/` with a format prefix: `int-train.csv`, `int-test.csv`, `int-eval.csv` 

## Data Generation Rules
- **Train (100K)**: includes all integers `0..1000`, plus random integers in `0..999999999` 
- **Test (10K)**: random integers in `0..999999999`, excluding any included in train dataset.
- **Eval (10,000 total)**: random integers in `0..999999999`, excluding integers in train or test.

## How to Generate
- Default output directory: `python sandbox.py --generate-data`
- Custom output directory: `python sandbox.py --generate-data --output-dir data`

## Spelling Rules
Numbers use English words with spaces, including "and" in the British style:
- `123` -> "one hundred and twenty three"
- `1001` -> "one thousand and one"
The label is the length of this string, counting letters and spaces.
