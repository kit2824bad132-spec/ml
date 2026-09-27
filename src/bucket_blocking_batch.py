
import duckdb
import os

TEST_DIR = "dataset/test"
OUTPUT_DIR = "output"
DB_FILE = "dataset/blocking.duckdb"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 5000
MAX_BLOCK_CANDIDATES = 250

con = duckdb.connect(DB_FILE)
con.execute("SET memory_limit='10GB'")
con.execute("SET threads=6")
con.execute("SET preserve_insertion_order=false")

print("Loading Source2 + Source3...")

con.execute(f"""
CREATE OR REPLACE TABLE source23 AS
SELECT
    entity_id,
    lower(trim(regexp_replace(coalesce(business_name,''),'[^[:alnum:] ]',' ','g'))) AS name_norm,
    lower(trim(coalesce(country,''))) AS country,
    left(replace(lower(trim(regexp_replace(coalesce(business_name,''),'[^[:alnum:] ]',' ','g'))),' ',''),6) AS prefix6,
    split_part(lower(trim(regexp_replace(coalesce(business_name,''),'[^[:alnum:] ]',' ','g'))),' ',1) AS token
FROM read_csv_auto('{TEST_DIR}/test_source2.tsv', delim='\\t', all_varchar=true)

UNION ALL

SELECT
    entity_id,
    lower(trim(regexp_replace(coalesce(business_name,''),'[^[:alnum:] ]',' ','g'))),
    lower(trim(coalesce(country,''))),
    left(replace(lower(trim(regexp_replace(coalesce(business_name,''),'[^[:alnum:] ]',' ','g'))),' ',''),6),
    split_part(lower(trim(regexp_replace(coalesce(business_name,''),'[^[:alnum:] ]',' ','g'))),' ',1)
FROM read_csv_auto('{TEST_DIR}/test_source3.tsv', delim='\\t', all_varchar=true);
""")

# Create indexes
con.execute("CREATE INDEX IF NOT EXISTS idx_exact ON source23(country,name_norm);")
con.execute("CREATE INDEX IF NOT EXISTS idx_prefix ON source23(country,prefix6);")
con.execute("CREATE INDEX IF NOT EXISTS idx_token ON source23(country,token);")

con.execute(f"""
CREATE OR REPLACE TABLE allowed_prefix6 AS
SELECT country, prefix6
FROM source23
WHERE length(prefix6) = 6
GROUP BY country, prefix6
HAVING COUNT(*) <= {MAX_BLOCK_CANDIDATES}
""")

con.execute(f"""
CREATE OR REPLACE TABLE allowed_token AS
SELECT country, token
FROM source23
WHERE length(token) >= 5
GROUP BY country, token
HAVING COUNT(*) <= {MAX_BLOCK_CANDIDATES}
""")

print("Loading Source1...")

con.execute(f"""
CREATE OR REPLACE TABLE source1 AS
SELECT
    entity_id,
    lower(trim(regexp_replace(coalesce(business_name,''),'[^[:alnum:] ]',' ','g'))) AS name_norm,
    lower(trim(coalesce(country,''))) AS country,
    left(replace(lower(trim(regexp_replace(coalesce(business_name,''),'[^[:alnum:] ]',' ','g'))),' ',''),6) AS prefix6,
    split_part(lower(trim(regexp_replace(coalesce(business_name,''),'[^[:alnum:] ]',' ','g'))),' ',1) AS token
FROM read_csv_auto('{TEST_DIR}/test_source1.tsv', delim='\\t', all_varchar=true);
""")

total_s1 = con.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
print(f"Total S1: {total_s1:,}")

output_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

with open(output_file, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")

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
        WITH matches AS (

            SELECT b.entity_id source1_entity_id,r.entity_id candidate_entity_id
            FROM batch_df b
            JOIN source23 r
            ON b.country=r.country
            AND b.name_norm=r.name_norm

            UNION

            SELECT b.entity_id,r.entity_id
            FROM batch_df b
            JOIN allowed_prefix6 k
            ON b.country=k.country
            AND b.prefix6=k.prefix6
            JOIN source23 r
            ON b.country=r.country
            AND b.prefix6=r.prefix6

            UNION

            SELECT b.entity_id,r.entity_id
            FROM batch_df b
            JOIN allowed_token k
            ON b.country=k.country
            AND b.token=k.token
            JOIN source23 r
            ON b.country=r.country
            AND b.token=r.token

        )

        SELECT
            b.entity_id source1_entity_id,
            COALESCE(
                string_agg(DISTINCT m.candidate_entity_id,',' ORDER BY m.candidate_entity_id),
                ''
            ) candidate_entity_ids
        FROM batch_df b
        LEFT JOIN matches m
        ON b.entity_id=m.source1_entity_id
        GROUP BY b.entity_id
        ORDER BY b.entity_id
    """).fetchdf()

    result.to_csv(
        output_file,
        sep="\t",
        index=False,
        mode="a",
        header=False
    )

    processed += len(batch)

    print(f"Processed {processed:,}/{total_s1:,}")

print("\nDONE!")
print("Output:", output_file)

con.close()