import os

import duckdb


TEST_DIR = os.environ.get("ER_TEST_DIR", "dataset/test")
OUTPUT_DIR = "output"
DB_FILE = os.environ.get("ER_BLOCKING_DB", "lightgbm_blocking.duckdb")
OUTPUT_FILE = os.environ.get(
    "ER_CANDIDATE_OUTPUT",
    os.path.join(OUTPUT_DIR, "candidate_pairs_lightgbm.tsv"),
)
BATCH_SIZE = 5000
MAX_BLOCK_CANDIDATES = 250

os.makedirs(OUTPUT_DIR, exist_ok=True)
con = duckdb.connect(DB_FILE)
con.execute("SET memory_limit='10GB'")
con.execute("SET threads=6")
con.execute("SET preserve_insertion_order=false")
con.execute("""
CREATE OR REPLACE MACRO normalize_block_text(input_text) AS
lower(trim(regexp_replace(
    regexp_replace(coalesce(input_text, ''), '[^[:alnum:] ]', ' ', 'g'),
    ' +', ' ', 'g'
)))
""")

print("Loading Source 2 and Source 3...")
con.execute(f"""
CREATE OR REPLACE TABLE source23 AS
SELECT
    entity_id,
    lower(trim(coalesce(country, ''))) AS country,
    normalize_block_text(business_name) AS name_norm,
    normalize_block_text(business_address) AS address_norm,
    left(replace(normalize_block_text(business_name), ' ', ''), 6) AS prefix6,
    split_part(normalize_block_text(business_name), ' ', 1) AS first_token
FROM read_csv_auto('{TEST_DIR}/test_source2.tsv', delim='\\t', all_varchar=true)
UNION ALL
SELECT
    entity_id,
    lower(trim(coalesce(country, ''))),
    normalize_block_text(business_name),
    normalize_block_text(business_address),
    left(replace(normalize_block_text(business_name), ' ', ''), 6),
    split_part(normalize_block_text(business_name), ' ', 1)
FROM read_csv_auto('{TEST_DIR}/test_source3.tsv', delim='\\t', all_varchar=true)
""")

print("Preparing bounded blocks...")
con.execute(f"""
CREATE OR REPLACE TABLE allowed_exact AS
SELECT country, name_norm
FROM source23
WHERE name_norm <> ''
GROUP BY country, name_norm
HAVING COUNT(*) <= {MAX_BLOCK_CANDIDATES}
""")
con.execute(f"""
CREATE OR REPLACE TABLE allowed_prefix6 AS
SELECT country, prefix6
FROM source23
WHERE length(prefix6) = 6
GROUP BY country, prefix6
HAVING COUNT(*) <= {MAX_BLOCK_CANDIDATES}
""")
con.execute(f"""
CREATE OR REPLACE TABLE allowed_first_token AS
SELECT country, first_token
FROM source23
WHERE length(first_token) >= 5
GROUP BY country, first_token
HAVING COUNT(*) <= {MAX_BLOCK_CANDIDATES}
""")
con.execute("""
CREATE OR REPLACE TABLE source23_postal AS
SELECT DISTINCT entity_id, country, code AS postal_code
FROM source23, UNNEST(string_split(address_norm, ' ')) u(code)
WHERE regexp_full_match(code, '[0-9]{5,6}')
""")
con.execute(f"""
CREATE OR REPLACE TABLE allowed_postal AS
SELECT country, postal_code
FROM source23_postal
GROUP BY country, postal_code
HAVING COUNT(*) <= {MAX_BLOCK_CANDIDATES}
""")

print("Loading Source 1...")
con.execute(f"""
CREATE OR REPLACE TABLE source1 AS
SELECT
    entity_id,
    lower(trim(coalesce(country, ''))) AS country,
    normalize_block_text(business_name) AS name_norm,
    normalize_block_text(business_address) AS address_norm,
    left(replace(normalize_block_text(business_name), ' ', ''), 6) AS prefix6,
    split_part(normalize_block_text(business_name), ' ', 1) AS first_token
FROM read_csv_auto('{TEST_DIR}/test_source1.tsv', delim='\\t', all_varchar=true)
""")

total_s1 = con.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
print(f"Source 1 records: {total_s1:,}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as output:
    output.write("source1_entity_id\tcandidate_entity_ids\n")

processed = 0
while processed < total_s1:
    batch = con.execute(f"""
        SELECT *
        FROM source1
        ORDER BY entity_id
        LIMIT {BATCH_SIZE}
        OFFSET {processed}
    """).fetchdf()
    con.register("batch_df", batch)

    result = con.execute("""
        WITH batch_postal AS (
            SELECT DISTINCT b.entity_id, b.country, u.code AS postal_code
            FROM batch_df b,
                 UNNEST(string_split(b.address_norm, ' ')) u(code)
            WHERE regexp_full_match(u.code, '[0-9]{5,6}')
        ),
        matches AS (
            SELECT b.entity_id AS source1_entity_id,
                   r.entity_id AS candidate_entity_id
            FROM batch_df b
            JOIN allowed_exact k
              ON b.country = k.country AND b.name_norm = k.name_norm
            JOIN source23 r
              ON b.country = r.country AND b.name_norm = r.name_norm

            UNION

            SELECT b.entity_id, r.entity_id
            FROM batch_df b
            JOIN allowed_prefix6 k
              ON b.country = k.country AND b.prefix6 = k.prefix6
            JOIN source23 r
              ON b.country = r.country AND b.prefix6 = r.prefix6

            UNION

            SELECT b.entity_id, r.entity_id
            FROM batch_df b
            JOIN allowed_first_token k
              ON b.country = k.country AND b.first_token = k.first_token
            JOIN source23 r
              ON b.country = r.country AND b.first_token = r.first_token

            UNION

            SELECT bp.entity_id, p.entity_id
            FROM batch_postal bp
            JOIN allowed_postal k
              ON bp.country = k.country AND bp.postal_code = k.postal_code
            JOIN source23_postal p
              ON bp.country = p.country AND bp.postal_code = p.postal_code
        )
        SELECT b.entity_id AS source1_entity_id,
               COALESCE(
                   string_agg(DISTINCT m.candidate_entity_id, ',' ORDER BY m.candidate_entity_id),
                   ''
               ) AS candidate_entity_ids
        FROM batch_df b
        LEFT JOIN matches m ON b.entity_id = m.source1_entity_id
        GROUP BY b.entity_id
        ORDER BY b.entity_id
    """).fetchdf()
    result.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        mode="a",
        header=False,
    )
    processed += len(batch)
    print(f"Processed {processed:,}/{total_s1:,}")

con.close()
print(f"Candidate output: {OUTPUT_FILE}")