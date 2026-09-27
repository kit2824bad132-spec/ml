import re
import unicodedata
import pandas as pd


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.lower()

    cleaned = []

    for char in value:
        category = unicodedata.category(char)

        if (
            category.startswith("L")
            or category.startswith("N")
            or category.startswith("M")
            or char.isspace()
        ):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    value = "".join(cleaned)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_country(value):
    if pd.isna(value):
        return ""

    return str(value).strip().lower()