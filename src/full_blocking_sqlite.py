import os
import re
import sqlite3
import pandas as pd

TEST_DIR = "dataset/test"
OUTPUT_DIR = "output"

S1_FILE = os.path.join(TEST_DIR, "test_source1.tsv")
S2_FILE = os.path.join(TEST_DIR, "test_source2.tsv")
S3_FILE = os.path.join(TEST_DIR, "test_source3.tsv")

DB_FILE = "dataset/test_blocking.db"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    cleaned = []

    for char in value:

        if char.isalnum() or char.isspace():
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


LEGAL_SUFFIXES = {
    "pvt",
    "private",
    "limited",
    "ltd",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "plc",
    "llp"
}


def remove_legal_suffixes(name):

    tokens = name.split()

    tokens = [
        token
        for token in tokens
        if token not in LEGAL_SUFFIXES
    ]

    return " ".join(tokens)


def make_blocks(name, country):

    if not name:
        return []

    base = remove_legal_suffixes(name)

    blocks = []

    # Exact normalized name
    blocks.append(
        ("exact", country + "|" + name)
    )

    # Name without legal suffix
    if base:
        blocks.append(
            ("base", country + "|" + base)
        )

    # First 4 compact characters
    compact = base.replace(" ", "")

    if len(compact) >= 4:

        blocks.append(
            ("prefix4", country + "|" + compact[:4])
        )

    # First token only when reasonably distinctive
    tokens = base.split()

    if tokens:

        first = tokens[0]

        if len(first) >= 5:

            blocks.append(
                ("first", country + "|" + first)
            )

    return blocks


# ============================================================
# CREATE SQLITE DATABASE
# ============================================================

def create_database():

    if os.path.exists(DB_FILE):

        print("Existing database found.")
        print("Deleting:", DB_FILE)

        os.remove(DB_FILE)

    print()
    print("Creating SQLite database...")

    conn = sqlite3.connect(DB_FILE)

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")

    conn.execute("""
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY
        )
    """)

    conn.execute("""
        CREATE TABLE blocks (
            block_type TEXT,
            block_key TEXT,
            entity_id TEXT
        )
    """)

    conn.execute("""
        CREATE INDEX idx_blocks
        ON blocks(block_type, block_key)
    """)

    conn.commit()

    return conn


# ============================================================
# INDEX S2 / S3
# ============================================================

def index_source(conn, source_file):

    print()
    print("Indexing:", source_file)

    total = 0

    for chunk in pd.read_csv(
        source_file,
        sep="\t",
        chunksize=50000,
        dtype=str,
        keep_default_na=False
    ):

        records = []
        block_rows = []

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id

            name = normalize_text(
                row.business_name
            )

            country = normalize_country(
                row.country
            )

            records.append(
                (entity_id,)
            )

            for block_type, block_key in make_blocks(
                name,
                country
            ):

                block_rows.append(
                    (
                        block_type,
                        block_key,
                        entity_id
                    )
                )

        conn.executemany(
            "INSERT OR IGNORE INTO records VALUES (?)",
            records
        )

        conn.executemany(
            "INSERT INTO blocks VALUES (?, ?, ?)",
            block_rows
        )

        conn.commit()

        total += len(chunk)

        print(
            "Processed:",
            total
        )

    print(
        "Finished:",
        source_file
    )


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(conn):

    print()
    print("==========================================")
    print("GENERATING TEST CANDIDATES")
    print("==========================================")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    first_write = True

    processed = 0

    total_candidates = 0

    max_candidates = 0

    zero_candidates = 0

    for chunk in pd.read_csv(
        S1_FILE,
        sep="\t",
        chunksize=10000,
        dtype=str,
        keep_default_na=False
    ):

        output_rows = []

        for row in chunk.itertuples(index=False):

            s1_id = row.entity_id

            name = normalize_text(
                row.business_name
            )

            country = normalize_country(
                row.country
            )

            candidates = set()

            blocks = make_blocks(
                name,
                country
            )

            for block_type, block_key in blocks:

                cursor = conn.execute(
                    """
                    SELECT entity_id
                    FROM blocks
                    WHERE block_type = ?
                    AND block_key = ?
                    """,
                    (block_type, block_key)
                )

                for result in cursor:

                    candidates.add(
                        result[0]
                    )

            candidate_count = len(candidates)

            total_candidates += candidate_count

            if candidate_count == 0:
                zero_candidates += 1

            if candidate_count > max_candidates:
                max_candidates = candidate_count

            output_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(
                    sorted(candidates)
                )
            })

        output_df = pd.DataFrame(
            output_rows
        )

        output_df.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write
        )

        first_write = False

        processed += len(chunk)

        print(
            f"S1 processed: {processed:,} | "
            f"average candidates: "
            f"{total_candidates / processed:.2f} | "
            f"max: {max_candidates:,} | "
            f"zero: {zero_candidates:,}"
        )

    print()
    print("==========================================")
    print("BLOCKING COMPLETE")
    print("==========================================")

    print(
        f"S1 records: {processed:,}"
    )

    print(
        f"Total candidate links: "
        f"{total_candidates:,}"
    )

    print(
        f"Average candidates/S1: "
        f"{total_candidates / processed:.2f}"
    )

    print(
        f"Maximum candidates for one S1: "
        f"{max_candidates:,}"
    )

    print(
        f"S1 with zero candidates: "
        f"{zero_candidates:,}"
    )

    print()
    print(
        "Candidate file:",
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    conn = create_database()

    try:

        index_source(
            conn,
            S2_FILE
        )

        index_source(
            conn,
            S3_FILE
        )

        generate_candidates(
            conn
        )

    finally:

        conn.close()