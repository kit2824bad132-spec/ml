$ErrorActionPreference = "Stop"

Write-Host "=================================================="
Write-Host "AMAZON ML CHALLENGE 2026 - FULL TEST PIPELINE"
Write-Host "=================================================="

$runId = Get-Date -Format yyyyMMdd_HHmmss
$candidatePath = "output/candidate_pairs_final_improved_$runId.tsv"
$blockingDb = "lightgbm_blocking_final_improved_$runId.duckdb"
$resultPath = "output/matching_results_final_improved_$runId.tsv"
$inferenceDb = "clean_final_inference_$runId.duckdb"

while ((Test-Path $candidatePath) -or (Test-Path $blockingDb) -or (Test-Path $resultPath) -or (Test-Path $inferenceDb)) {
    $runId = Get-Date -Format yyyyMMdd_HHmmss_fff
    $candidatePath = "output/candidate_pairs_final_improved_$runId.tsv"
    $blockingDb = "lightgbm_blocking_final_improved_$runId.duckdb"
    $resultPath = "output/matching_results_final_improved_$runId.tsv"
    $inferenceDb = "clean_final_inference_$runId.duckdb"
}

Write-Host "Run ID: $runId"
Write-Host "Candidate Path: $candidatePath"
Write-Host "Blocking DB: $blockingDb"
Write-Host "Result Path: $resultPath"
Write-Host "Inference DB: $inferenceDb"
Write-Host "Model File: dataset/train/clean_validation_model_improved.pkl"

Write-Host "`n[PHASE 1/2] RUNNING TEST BLOCKING..."
$env:ER_BLOCKING_DB = $blockingDb
$env:ER_CANDIDATE_OUTPUT = $candidatePath
$env:ER_TEST_DIR = "dataset/test"

& .\.venv\Scripts\python.exe src\bucket_blocking_lightgbm.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Blocking failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`nConfirming candidate file..."
if (-not (Test-Path $candidatePath)) {
    Write-Error "Candidate file was not created: $candidatePath"
    exit 1
}

$firstLine = Get-Content -Path $candidatePath -TotalCount 1
Write-Host "Candidate header: $firstLine"
if ($firstLine -ne "source1_entity_id`tcandidate_entity_ids") {
    Write-Error "Unexpected header in candidate file: $firstLine"
    exit 1
}
Write-Host "Candidate generation successful!"

Write-Host "`n[PHASE 2/2] RUNNING INFERENCE..."
$env:ER_CANDIDATE_FILE = $candidatePath
$env:ER_MODEL_FILE = "dataset/train/clean_validation_model_improved.pkl"
$env:ER_MATCHING_OUTPUT = $resultPath
$env:ER_INFERENCE_DB = $inferenceDb

& .\.venv\Scripts\python.exe run_inference_duckdb.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Inference failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`n=================================================="
Write-Host "PIPELINE COMPLETED - RUNNING VALIDATION"
Write-Host "=================================================="

& .\.venv\Scripts\python.exe -c "
import sys, os

result_path = '$resultPath'
candidate_path = '$candidatePath'
s1_path = 'dataset/test/test_source1.tsv'

if not os.path.exists(result_path):
    print(f'ERROR: Result file does not exist: {result_path}')
    sys.exit(1)

with open(result_path, 'r', encoding='utf-8') as f:
    header = f.readline()
    if header != 'source1_entity_id\tmatched_entity_ids\n':
        print(f'ERROR: Invalid header in result: {repr(header)}')
        sys.exit(1)

print('Streaming line counts...')
total_result_rows = 0
nonempty_matches = 0
with open(result_path, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        total_result_rows += 1
        parts = line.rstrip('\r\n').split('\t')
        if len(parts) > 1 and parts[1]:
            nonempty_matches += 1

print('Counting test_source1 records...')
s1_rows = 0
with open(s1_path, 'r', encoding='utf-8') as f:
    f.readline()
    for _ in f:
        s1_rows += 1

print('Counting candidate pairs/links...')
cand_s1_count = 0
cand_link_count = 0
with open(candidate_path, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        cand_s1_count += 1
        parts = line.rstrip('\r\n').split('\t')
        if len(parts) > 1 and parts[1]:
            cand_link_count += len(parts[1].split(','))

print('=== VALIDATION REPORT ===')
print(f'Candidate File: {candidate_path} ({os.path.getsize(candidate_path):,} bytes)')
print(f'Candidate S1 Count: {cand_s1_count:,}')
print(f'Candidate Total Links: {cand_link_count:,}')
print(f'Result File: {result_path} ({os.path.getsize(result_path):,} bytes)')
print(f'Expected S1 Rows: {s1_rows:,}')
print(f'Result Rows: {total_result_rows:,}')
print(f'Nonempty Matches: {nonempty_matches:,} ({nonempty_matches/total_result_rows*100.2:.2f}%)')
print(f'Empty Matches: {total_result_rows - nonempty_matches:,}')

if total_result_rows != s1_rows:
    print(f'ERROR: Row count mismatch! Got {total_result_rows}, expected {s1_rows}')
    sys.exit(1)

print('VALIDATION STATUS: PASS')
"

Write-Host "All finished successfully!"
