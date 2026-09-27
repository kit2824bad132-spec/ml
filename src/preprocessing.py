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
    return value


def normalize_country(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()



LEGAL_SUFFIX_REGEX = re.compile(
    r"\b(private limited|pvt ltd|pvt|ltd|limited|llc|inc|corp|corporation|co|company|sarl|sa|gmbh|bv|sl|srl|spa|plc|holdings|group|enterprise|enterprises|services)\b",
    re.IGNORECASE,
)


def strip_legal_suffixes(value):
    if pd.isna(value):
        return ""
    value = str(value)
    value = LEGAL_SUFFIX_REGEX.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def number_match_score(addr1, addr2):
    if pd.isna(addr1) or pd.isna(addr2):
        return 0.0
    nums1 = set(re.findall(r"\b\d+\b", str(addr1)))
    nums2 = set(re.findall(r"\b\d+\b", str(addr2)))
    if not nums1 or not nums2:
        return 0.0
    if nums1 & nums2:
        return 1.0
    return -1.0