from preprocessing import normalize_text, normalize_country


examples = [
    "ABC Technologies Pvt. Ltd.",
    "ABC Technologies PRIVATE LIMITED",
    "Moyna's Coffee",
    "LLC Moncada Léarning Center",
    "राम मार्केटिंग प्राइवेट लिमिटेड"
]


print("TEXT NORMALIZATION")
print("=" * 50)

for text in examples:
    print("Original  :", text)
    print("Normalized:", normalize_text(text))
    print()


print("COUNTRY NORMALIZATION")
print("=" * 50)

countries = [
    "US",
    "us",
    "India",
    "FRANCE"
]

for country in countries:
    print(country, "->", normalize_country(country))