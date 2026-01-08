_ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def _words_1_to_99(value):
    if value < 20:
        return _ONES[value]
    tens = (value // 10) * 10
    ones = value % 10
    if ones == 0:
        return _TENS[tens]
    return f"{_TENS[tens]} {_ONES[ones]}"


def _words_1_to_999(value):
    if value < 100:
        return _words_1_to_99(value)
    hundreds = value // 100
    remainder = value % 100
    if remainder == 0:
        return f"{_ONES[hundreds]} hundred"
    return f"{_ONES[hundreds]} hundred and {_words_1_to_99(remainder)}"


def number_to_words(value):
    if value == 0:
        return _ONES[0]
    parts = []
    remainder = value
    if remainder >= 1_000_000:
        millions = remainder // 1_000_000
        remainder %= 1_000_000
        parts.append(f"{_words_1_to_999(millions)} million")
    if remainder >= 1_000:
        thousands = remainder // 1_000
        remainder %= 1_000
        parts.append(f"{_words_1_to_999(thousands)} thousand")
    if remainder > 0:
        if parts and remainder < 100:
            parts.append(f"and {_words_1_to_99(remainder)}")
        else:
            parts.append(_words_1_to_999(remainder))
    return " ".join(parts)
