import duckdb
import os

TEST_DIR = "dataset/test"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

con = duckdb.connect("dataset/blocking.duckdb")

# Use available RAM efficiently
con.execute("SET memory_limit='12GB'")
con.execute("SET threads=8")

print("Loading S2 + S3...")

con.execute(f"""
CREATE OR REPLACE TABLE source23 AS

SELECT
entity_id,
lower(trim(regexp_replace(coalesce(business_name,''),'[^[:alnum:] ]',' ','g'))) name_norm,
lower(trim(regexp_replace(coalesce(business_address,''),'[^[:alnum:] ]',' ','g'))) addr_norm,
lower(trim(coalesce(country,''))) country
FROM read_csv_auto('{TEST_DIR}/test_source2.tsv', delim='\\t', all_varchar=true)

UNION ALL

SELECT
entity_id,
lower(trim(regexp_replace(coalesce(business_name,''),'[^[:alnum:] ]',' ','g'))),
lower(trim(regexp_replace(coalesce(business_address,''),'[^[:alnum:] ]',' ','g'))),
lower(trim(coalesce(country,'')))
FROM read_csv_auto('{TEST_DIR}/test_source3.tsv', delim='\\t', all_varchar=true)
""")

print("Loading S1...")

con.execute(f"""
CREATE OR REPLACE TABLE source1 AS

SELECT
entity_id,
lower(trim(regexp_replace(coalesce(business_name,''),'[^[:alnum:] ]',' ','g'))) name_norm,
lower(trim(regexp_replace(coalesce(business_address,''),'[^[:alnum:] ]',' ','g'))) addr_norm,
lower(trim(coalesce(country,''))) country
FROM read_csv_auto('{TEST_DIR}/test_source1.tsv', delim='\\t', all_varchar=true)
""")

print("Creating block columns...")

con.execute("""
ALTER TABLE source23 ADD COLUMN prefix6 VARCHAR;
UPDATE source23 SET prefix6 = left(replace(name_norm,' ',''),6);

ALTER TABLE source1 ADD COLUMN prefix6 VARCHAR;
UPDATE source1 SET prefix6 = left(replace(name_norm,' ',''),6);

ALTER TABLE source23 ADD COLUMN token VARCHAR;
UPDATE source23 SET token = split_part(name_norm,' ',1);

ALTER TABLE source1 ADD COLUMN token VARCHAR;
UPDATE source1 SET token = split_part(name_norm,' ',1);

ALTER TABLE source23 ADD COLUMN addr6 VARCHAR;
UPDATE source23 SET addr6 = left(replace(addr_norm,' ',''),6);

ALTER TABLE source1 ADD COLUMN addr6 VARCHAR;
UPDATE source1 SET addr6 = left(replace(addr_norm,' ',''),6);
""")

print("Generating candidates...")

con.execute("""
CREATE OR REPLACE TABLE candidates AS

SELECT DISTINCT * FROM (

SELECT s.entity_id source1_entity_id,r.entity_id candidate_entity_id
FROM source1 s
JOIN source23 r
ON s.country=r.country
AND s.name_norm=r.name_norm

UNION

SELECT s.entity_id,r.entity_id
FROM source1 s
JOIN source23 r
ON s.country=r.country
AND s.prefix6=r.prefix6
AND length(s.prefix6)>=6

UNION

SELECT s.entity_id,r.entity_id
FROM source1 s
JOIN source23 r
ON s.country=r.country
AND s.token=r.token
AND length(s.token)>=5

UNION

SELECT s.entity_id,r.entity_id
FROM source1 s
JOIN source23 r
ON s.country=r.country
AND s.addr6=r.addr6
AND length(s.addr6)>=6

)
""")

print("Writing candidate_pairs.tsv...")

con.execute(f"""
COPY (

SELECT
s.entity_id source1_entity_id,

COALESCE(
string_agg(candidate_entity_id,',' ORDER BY candidate_entity_id),
''
) candidate_entity_ids

FROM source1 s

LEFT JOIN candidates c
ON s.entity_id=c.source1_entity_id

GROUP BY s.entity_id
ORDER BY s.entity_id

)

TO '{OUTPUT_DIR}/candidate_pairs.tsv'

WITH (HEADER, DELIMITER '\\t');
""")

stats=con.execute("""

SELECT
COUNT(*) total_links,
COUNT(DISTINCT source1_entity_id) matched_s1
FROM candidates

""").fetchone()

print("\nDONE")
print("Total candidate links:",f"{stats[0]:,}")
print("Matched S1:",f"{stats[1]:,}")
print("Output: output/candidate_pairs.tsv")

con.close()