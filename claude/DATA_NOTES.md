# Data Quality Notes

Observations from auditing the benchmark datasets and results.
Last updated: 2026-04-02.

---

## Eurorad (`csvs/Eurorad.csv`)

**228 rows, 61 columns** (after cleanup). Single ground truth column: `FinalDiagnosis`
(207/228 populated). Three corrupted alternate GT columns (`FinalDiagnosisv2`,
`FinalDiagnosisv2_Alhusain`, `FinalDiagnosisv3_Omar`) were removed.

### Qwen prediction row-shift (fixed)

Nine Qwen columns had predictions shifted by one row in the region case_id 19041-19087
(27 rows). The predictions were rotated: each row contained the prediction for the next
case, and the last row contained the prediction for the first case. Verified by confirming
that the predictions were not valid options for the assigned cases but were valid options
for the next case, across all 27 rows.

**Affected columns (all 3 runs):** Qwen3.5 35B (base), Qwen3.5 35B (fine-tuned),
Qwen3.5 9B (fine-tuned).

**Not affected:** Qwen3.5 27B, Qwen3.5 9B (base) -- these were correctly aligned.

**Root cause:** Spreadsheet paste error. The shifted columns were likely pasted with a
one-row offset in this region. The `FinalDiagnosisv2` column had the same shift, suggesting
it was derived from or created alongside the shifted Qwen predictions.

**Impact (row-shift fix only, without mojibake fix):**

| Model | Before fix | After fix (row-shift only) |
|-------|-----------|-----------|
| Qwen3.5 35B | 72.5% | **83.1%** |
| Qwen3.5 35B FT | 75.8% | **87.4%** |
| Qwen3.5 9B FT | 72.0% | **80.7%** |
| Qwen3.5 27B | 80.2% | 80.2% (unchanged) |
| Qwen3.5 9B | 76.3% | 76.3% (unchanged) |

**Final accuracy (both row-shift + mojibake fix via norm_text CP1252 recovery):**

| Model | Accuracy |
|-------|----------|
| Qwen3.5 35B | **84.1%** |
| Qwen3.5 35B FT | **87.9%** |
| Qwen3.5 9B FT | **81.2%** |
| Qwen3.5 27B | 80.2% (unchanged) |
| Qwen3.5 9B | 76.3% (unchanged) |

The difference between the two "after fix" columns is the CP1252 mojibake recovery in
`norm_text()`, which rescues ~1-2% additional correct answers for affected Qwen models.
Qwen 27B and 9B base are unaffected by the row-shift but still benefit slightly from the
mojibake fix (their scores happen to round to the same values).

**Status:** Fixed in `csvs/final_csvs/Eurorad.csv`. The git-tracked `csvs/Eurorad.csv`
still contains the unfixed data; `csvs/updated_csvs/Eurorad.csv` also has unfixed Qwen
data. Only `csvs/final_csvs/Eurorad.csv` has both fixes applied.

### Corrupted ground truth columns (removed)

Three alternate GT columns were removed from the CSV:

| Column | Issues |
|--------|--------|
| `FinalDiagnosisv2` | 17 rows CP1252 mojibake + 27 rows shifted (same shift as Qwen predictions). Previous summary sheet numbers (e.g., Qwen 35B = 83.9%) were computed against this column, inflating Qwen scores while deflating clean models (GPT-5.2 dropped from 90.8% to 77.8% when scored against v2). |
| `FinalDiagnosisv2_Alhusain` | Identical to v2. |
| `FinalDiagnosisv3_Omar` | 18 rows mojibake + 7 rows shifted (case_id 19089-19101) + case 19145 had Section value leaked into diagnosis field. |

### Qwen 3.5 mojibake (all variants)

Qwen model outputs contain CP1252 mojibake on ~20 rows per run. The input data
(`DifferentialDiagnosisList`) is clean -- corruption happens in the Qwen tokenizer/decoding
pipeline.

**Impact:** ~15-18 correct answers scored wrong per run before fix.
**Status:** Fixed in `norm_text()` / `normalize_text()` via CP1252 recovery step.
Clean models (GPT-5, OSS, DeepSeek, Gemini) are unaffected.

### Outlier cases (xlsx sheet "Outlier cases for Eurorad")

4 cases flagged as problematic for exact string matching:

| case_id | Issue |
|---------|-------|
| 18836 | OEIS complex -- very long multi-system congenital anomaly label |
| 18951 | Ground truth contains citation bracket: `Biventricular endomyocardial fibrosis [4,6]` |
| 19158 | Very specific molecular subtype with molecular fusion notation |
| 19163 | Minor formatting variant: `, type 1` suffix |

### 21 rows with no ground truth

228 total rows, but only 207 have `FinalDiagnosis` populated. The 21 blank rows should
be investigated -- they may be cases excluded for quality reasons or simply missing data.

---

## Ophthalmology (`csvs/Ophthalmology.csv`, xlsx "Ophthalmology" sheet)

### Data structure

- 1003 rows total, only **130 have ground truth** (`GT` column). The remaining 873 rows
  are entirely empty (no Question, no GT).
- Answers are multi-select MCQ letters (e.g., `A`, `BD`, `ABCFGH`).
- GT column has 27 unique answer patterns.

### Encoding: Clean

Zero mojibake across all 48 model columns and GT. Letter-based answers avoid the Unicode
issues seen in Eurorad.

### Whitespace in multi-select answers

`gpt-5.1-1113` (all 3 runs) has `ABCFG H` instead of `ABCFGH` on row 119. The
`parse_choice` regex required contiguous letters, dropping the `H`.

**Status:** Fixed -- `parse_choice` now collapses whitespace before matching.

### Model coverage

| Model group | Columns | Runs |
|-------------|---------|------|
| GPT-5 (0807, 5.1-1113, 5.2-1211) | 9 | 3 each |
| GPT-5-mini-0807 | 3 | 3 |
| o4-mini | 3 | 3 |
| Gemini 3.1 Pro | 3 | 3 |
| DeepSeek R1 0528 | 3 | 3 |
| OSS-20B (L/M/H) | 9 | 3 each |
| OSS-120B (L/M/H) | 9 | 3 each |
| Qwen 3.5 (9B, 27B, 35B) | 9 | 3 each |

No fine-tuned Qwen columns in ophthalmology (unlike Eurorad).

---

## NMED Diagnosis (`csvs/NMED_Diagnosis.csv`, `csvs/NMED_Diagnosis_3x.csv`)

### Data structure

- **720 data rows** out of 1008 total. The remaining 288 rows are empty or contain
  spreadsheet artifacts (summary stats, `Loading...`, column headers repeated as values).
- Each row is one (disease, source_model) combination. The `Disease_description` column
  contains the source model's output. The score columns are **evaluator** Likert scores
  (1-5) of that output.
- `Model` column: the model that *generated* the response being scored (GPT-4o, GPT-3.5,
  GPT-4, Gem2FTE, DeepSeek-R1, DeepSeek-V3). Not the evaluator.
- 5 clinical specialties: Internal Medicine, Surgery, Pediatrics, Neurology, Gynecology.
- 5 diseases have 2 distinct patient cases each (different descriptions, same disease):
  Acute appendicitis, Cluster headache, Endometriosis, Febrile seizure, Peripheral artery
  disease. This is intentional, not a data error.

### Encoding: Clean

Zero mojibake in any column. Likert scores avoid Unicode issues entirely.

### Spreadsheet artifacts in trailing rows

Rows 720+ contain non-data content leaked from the spreadsheet:
- **`Loading...`** values (8 rows) -- broken formula references, likely from cells that
  hadn't finished computing when the sheet was exported
- **Column headers as values** (row 731) -- e.g., `gpt5-0807` in the `gpt5-0807-m1` column
- **Summary statistics labels** (rows 734-739) -- `MEAN ABS ERR w/ IQR`, `MED ABS ERR w/ IQR`,
  `MEAN w/ 95% CI`, `MEAN ABS ERR w/ CI`, `Wilcoxon`. These are row labels for aggregation
  formulas whose values didn't export (show as `Loading...`).
- **Stray numeric value** (row 725) -- `0.7663194444` in gpt5-0807-m1 column with no
  metadata. Likely a formula result that leaked.

**Impact:** Any analysis that doesn't filter to `Clinical specialty.notna()` will include
junk rows. All score columns have non-numeric values in the trailing rows.

### Frequency column: inconsistent casing

60 rows use lowercase (`frequent`, `less frequent`, `rare`) while the rest use title case
(`Frequent`, `Less Frequent`, `Rare`). The lowercase rows span GPT-4o, Gem2FTE, DeepSeek-R1,
and DeepSeek-V3. Any groupby on Frequency will split these into separate categories unless
normalized.

### Score characteristics

- All scores within [1.0, 5.0] range (valid Likert scale).
- `HumanEvalScore` uses quarter-point steps (1.5, 2.25, 2.5, 2.75, 3.25, ..., 4.75).
  Mean: 4.46.
- Model evaluator scores use half-point steps (1.5, 2.5, 3.5, 4.5) with varying
  frequency. `o4-mini` is the most integer-heavy; `gpt5-0807` and `oss120b` produce
  the most half-point scores.
- `NMED_Diagnosis_3x.csv` has the full model lineup with 3 runs each (48 score columns +
  Qwen variants). All 720 data rows have complete scores across all evaluator columns.

### Model coverage

| File | Score columns | Notes |
|------|--------------|-------|
| `NMED_Diagnosis.csv` | 10 (HumanEval + 9 models, 1 run each) | Base file |
| `NMED_Diagnosis_3x.csv` | 49 (HumanEval + 48 model runs) | Full 3-run lineup incl. Qwen |
| `NMED_Diagnosis_beams.csv` | 11 (HumanEval + gpt5 + oss20b beams at 5/9/11) | Beam search variants |

---

## NMED Treatment (`csvs/NMED_Treatment.csv`, `csvs/NMED_Treatment_3x.csv`)

### Data structure

- **595 data rows** out of 999 total. 404 non-data rows (same spreadsheet artifact issue
  as Diagnosis).
- Same row structure as Diagnosis: each row = (disease, source_model) scored by evaluators.
- `Model` column: GPT-4o, GPT-3.5, GPT-4, Gem2FTE, DeepSeek-R1. (No DeepSeek-V3, unlike
  Diagnosis.)
- Same 5 clinical specialties and disease overlap pattern.

### Encoding: Clean

Zero mojibake.

### Spreadsheet artifacts

Same pattern as Diagnosis:
- `Loading...` values in trailing rows
- `#NAME?` error in row 604 (`gpt5-0807-m1` column) -- broken spreadsheet formula
- Summary stat labels (`MEAN ABS ERR`, `Wilcoxon`, etc.) in HumanEvalScore column
- 2 fully-empty `Unnamed: 15` / `Unnamed: 16` columns (artifact of trailing commas in CSV)

### Frequency column: same casing issue

45 rows use lowercase vs title case. Same pattern as Diagnosis.

### Score characteristics

- All scores within [1.0, 5.0] range.
- `HumanEvalScore` mean: 4.41.
- `gpt-5.2-1211` in Treatment 3x has the highest non-integer rate (~42% of scores are
  half-point), vs. `o4-mini` and `gemini-3.1-pro` which are almost entirely integers.

### Missing data in Treatment 3x

Most evaluator columns have all 595 data rows populated, but:
- `deepseek-r1-0528-v2`: 582/595 (13 missing)
- `deepseek-r1-0528-v3`: 585/595 (10 missing)
- `gemini-3.1-pro-m2`: 593/595 (2 missing)

These are likely API failures or timeouts during evaluation that weren't retried.

---

## xlsx structure (`csvs/Model_output.xlsx`)

26 sheets total. Key sheets:
- `Eurorad` -- primary results (228 x 64)
- `Ophthalmology` -- primary results (1003 x 53, 130 with GT)
- `NMED Diagnosis` / `NMED Treatment` -- primary NMED results
- `OmarOpthalmologyUpdated` -- aggregated accuracy with Wilson CIs, has broken `#REF!`
  formulas and `No Data` entries. Header at row 14, not row 0.
- `Outlier cases for Eurorad` -- 4 flagged cases
- `finetune-Eurorad` -- fine-tuning results
- `finetune-eval/` directory has per-checkpoint CSVs (steps 75-552, 3 runs each)

### Duplicate/stale sheets

`Copy of Eurorad`, `Copy of NMED Diagnosis`, `Copy of NMED Treatment` appear to be
backup copies. `Omar Opthalmology` is the predecessor of `OmarOpthalmologyUpdated`.
`Alif Eurorad` and `Alhusain NMED` are contributor-specific working copies.

### Column `6` in Ophthalmology

First column is named `6` (likely a row number artifact from the spreadsheet export).
Contains sequential float values (1.0, 2.0, ...). Not a meaningful column name.

---

## CSV directory structure

Three CSV directories exist with different states of the data:

| Directory | Eurorad | NMED | Ophthalmology | Notes |
|-----------|---------|------|---------------|-------|
| `csvs/` | 1179 rows, 61 cols, **no fixes**, no Gemma | 15/17 cols (base only, 1 run per model) | 53 cols, no Gemma | Git-tracked originals |
| `csvs/updated_csvs/` | 228 rows (test only), 68 cols, **no Qwen fix**, has Gemma | 57 cols (3x runs + Gemma) | 56 cols (+ Gemma) | Added Gemma 4, but Qwen row-shift unfixed |
| `csvs/final_csvs/` | 1179 rows, 65 cols, **both fixes**, has Gemma | 57 cols (copied from updated) | 56 cols (copied from updated) | **Use this for all analysis** |

**Key differences in Eurorad:**
- `csvs/` has float case_ids (`18812.0`), includes 972 train rows, has corrupted GT columns
- `csvs/updated_csvs/` has integer case_ids (`18812`), test-only (228 rows), has duplicate
  DeepSeek column headers (`deepseek r1 0528 v2` x3, deduplicated to `.1`, `.2` on read)
- `csvs/final_csvs/` has float case_ids (from git original), includes train rows, corrupted
  GT columns removed, Qwen row-shift fixed, Gemma merged with case_id normalization

**NMED and Ophthalmology** have no data quality issues. The `final_csvs/` versions are
direct copies of `updated_csvs/` (which have all model runs including Gemma 4).

---

## Gemma 4 (`csvs/updated_csvs/`, `csvs/final_csvs/`)

Added 2026-04-18. Two Gemma 4 models evaluated across all datasets.

### Eurorad column names
- `Gemma-4-26b-a4b` (single run, 26B model)
- `gemma-4-31b-M1`, `gemma-4-31b-M2`, `gemma-4-31b-M3` (3 runs, 31B model)

### NMED / Ophthalmology column names
- `gemma4 31B v1`, `gemma4 31B v2`, `gemma4 31B v3`

### Data quality: Clean

- Zero missing values across all datasets
- Zero mojibake (unlike Qwen, Gemma outputs are clean Unicode)
- Zero row-shift (verified: shifting in either direction drops accuracy to 0)
- Non-ASCII characters are legitimate: en-dash (U+2013), right single quote (U+2019),
  accented Latin letters (e, u, c). All handled by `norm_text()` normalization.
- Inter-run consistency: 98.1% on Eurorad (203/207 cases identical across 3 runs)

### Performance

| Dataset | Gemma 4 26B | Gemma 4 31B (majority vote) |
|---------|-------------|----------------------------|
| Eurorad | 77.3% (single run) | 86.5% |
| NMED Diagnosis | -- | See statistics output |
| NMED Treatment | -- | See statistics output |
| Ophthalmology | -- | See statistics output |
