# Changelog

All notable changes to this project are documented here.

---

## 2026-04-18

### Created: `csvs/final_csvs/` with corrected data

Built a clean dataset directory combining fixes from multiple sources:

- **Eurorad.csv**: Rebuilt from git-tracked original (`csvs/Eurorad.csv`) with three
  changes applied:
  1. Dropped corrupted GT columns (`FinalDiagnosisv2`, `v2_Alhusain`, `v3_Omar`)
  2. Applied Qwen row-shift fix (`np.roll(vals, +1)` on 9 columns, cases 19041-19087)
  3. Merged Gemma 4 columns from `csvs/updated_csvs/Eurorad.csv` with case_id
     normalization (stripping `.0` suffix from float-format IDs to match integer-format
     IDs in updated_csvs)
- **NMED_Diagnosis.csv, NMED_Treatment.csv**: Copied from `csvs/updated_csvs/` (57 cols,
  includes all 3x model runs + Gemma 4). No row-shift or mojibake issues in NMED data.
- **Ophthalmology.csv**: Copied from `csvs/updated_csvs/` (56 cols, includes Gemma 4).
  No data quality issues.

**Why not just use updated_csvs?** The updated_csvs Eurorad file still contains the
unfixed Qwen row-shift bug and the corrupted GT columns. The original git-tracked CSVs
lack Gemma 4 data. Only final_csvs has everything correct.

### Updated: All figure and statistics scripts point to `csvs/final_csvs/`

Changed 11 path references across 8 files from `csvs/updated_csvs/` to `csvs/final_csvs/`:
- `statistics/eurorad_stats.py`, `nmed_stats.py`, `export_csv.py`
- `figures/eurorad_accuracy_barplot.py`, `eurorad_finetune_barplot.py`,
  `eurorad_radar_plot.py`, `nmed_violin_plot.py`, `combined_figure.py`

### Fixed: Gemma 4 merge failure in Eurorad (case_id format mismatch)

**Problem:** First attempt at merging Gemma columns from updated_csvs into final_csvs
produced 0% accuracy because case_id formats differed: git original uses float strings
(`18812.0`) while updated_csvs uses integer strings (`18812`). The dictionary lookup
found zero matches.

**Fix:** Normalized case_ids via `str(int(float(cid)))` before matching. Gemma 4 31B
now correctly shows 86.5% majority-vote accuracy.

### Added: Gemma 4 31B to all figures and statistics

Added Gemma 4 31B model definitions to:
- `statistics/eurorad_stats.py`, `nmed_stats.py`, `export_csv.py`
- `figures/eurorad_accuracy_barplot.py`, `eurorad_radar_plot.py`, `eurorad_heatmap.py`,
  `nmed_violin_plot.py`, `combined_figure.py`
- `figures/palette.py` (color: `#F4A261` warm orange)

Regenerated all figures and `statistics/all_statistics.csv`.

### Verified: Final Eurorad accuracy numbers (both fixes applied)

With row-shift fix + CP1252 mojibake recovery in `norm_text()`:

| Model | Accuracy |
|-------|----------|
| Gemini 3.1 Pro | 91.8% |
| GPT-5.1 | 89.4% |
| Qwen3.5 35B FT | 87.9% |
| Gemma 4 31B | 86.5% |
| GPT-5-mini | 84.5% |
| Qwen3.5 35B | 84.1% |
| gpt-oss-120b (H) | 83.1% |
| DeepSeek-R1 | 81.6% |
| Qwen3.5 9B FT | 81.2% |
| gpt-oss-20b (H) | 80.7% |
| Qwen3.5 27B | 80.2% |
| Qwen3.5 9B | 76.3% |

Note: These differ slightly from the paper's reported numbers (e.g., paper says Qwen 35B
= 84.5%, GPT-5-mini = 84.1%) because the paper was written before the mojibake fix was
applied at evaluation time. The paper numbers came from row-shift-only fixes scored
without CP1252 recovery in `norm_text()`.

### Audited: Gemma 4 data quality (clean)

Verified Gemma 4 outputs have:
- Zero missing values (all 207 cases populated)
- Zero mojibake (no CP1252 encoding artifacts)
- Zero row-shift (shifting in either direction drops accuracy to 0)
- Non-ASCII chars are legitimate (en-dashes, curly quotes, accented letters)
- 98.1% inter-run agreement (203/207 cases identical across all 3 runs)

### EFS same-inode issue discovered

Both repo paths resolve to the same filesystem inode:
- `/mnt/custom-file-systems/efs/.../on-device-LLM/`
- `/home/sagemaker-user/user-default-efs/on-device-LLM/`

Copying files between these paths overwrites the source. Original CSVs were accidentally
overwritten this way; restored from git via `git show HEAD:results/<file>.csv`.

---

## 2026-04-02

### Fixed: Qwen prediction row-shift in Eurorad (cases 19041-19087)

**Problem:** Nine Qwen columns (35B base x3, 35B FT x3, 9B FT x3) had predictions
rotated by one row in the 27-row region case_id 19041-19087. Each row's prediction was
actually the answer for the next case. Verified by checking that predictions were not
valid options for the assigned cases but were valid options for the next case.

**Root cause:** Spreadsheet paste error. The same shift was present in the
`FinalDiagnosisv2` column, which was likely created from or alongside the Qwen results.

**Fix:** Rotated predictions back to correct positions. Unaffected columns: Qwen 27B,
Qwen 9B base (these were correctly aligned).

**Impact:** Qwen 35B: 72.5% -> 83.1%, Qwen 35B FT: 75.8% -> 87.4%, Qwen 9B FT: 72.0% -> 80.7%.

### Cleaned: Removed corrupted ground truth columns from Eurorad CSV

Dropped `FinalDiagnosisv2`, `FinalDiagnosisv2_Alhusain`, `FinalDiagnosisv3_Omar` from
`csvs/Eurorad.csv`. These had mojibake, row-shift bugs, and were the source of previously
reported inflated Qwen accuracy (83.9%) and deflated proprietary model accuracy.
`FinalDiagnosis` is the sole ground truth column (64 -> 61 columns).

### Fixed: Mojibake in text normalization (7-9% Qwen accuracy impact)

**Problem:** All Qwen 3.5 model outputs (9B, 27B, 35B, base + fine-tuned) contained
CP1252 mojibake -- Unicode characters like en-dash (`-`), right quote (`'`), `e`, `u`,
`c` were corrupted into multi-byte garbage (e.g., `Nail--patella` became `Naila EUR "patella`).

The `norm_text()` / `normalize_text()` functions across the codebase did not handle this,
causing ~15-18 correct Qwen answers per run to be scored as wrong (7-9% accuracy penalty
on 207 EuroRAD cases).

**Root cause:** The Qwen 3.5 tokenizer/decoding pipeline emits UTF-8 bytes that get
misinterpreted as CP1252 somewhere in the inference chain. The ground truth and candidate
options in `DifferentialDiagnosisList` are clean Unicode -- only the model output is corrupted.

**Fix:** Added a CP1252 recovery step (`s.encode("cp1252").decode("utf-8")`) at the top
of every text normalizer. The fix is a no-op on already-clean text (GPT-5, OSS, DeepSeek,
Gemini models are unaffected).

**Files changed (9):**
- `benchmarks/eurorad/hf_bench.py`
- `benchmarks/eurorad/gpt.py`
- `benchmarks/eurorad/openrouter.py`
- `benchmarks/eurorad/eval_finetune.py`
- `benchmarks/eurorad/eurorad_beams_hf.py`
- `benchmarks/eurorad/eurorad_beams_unsloth.py`
- `benchmarks/eurorad/eurorad_beams.py`
- `benchmarks/eurorad/oss20b_inference.py`
- `compute_eurorad_stats.py`

**Verified impact:**
- Qwen 3.5 35B: 63.3% -> 71.0% (v1), 65.7% -> 72.9% (v2, v3)
- Clean models: zero change

### Investigated: FinalDiagnosis ground truth columns in Eurorad

Determined that `FinalDiagnosis` is the correct ground truth column (clean Unicode, used
by all eval scripts). The alternative columns have known issues:

- `FinalDiagnosisv2` / `FinalDiagnosisv2_Alhusain`: 17 rows of mojibake + ~25 rows with
  a row-shift bug (case_id 19041-19087, each value is the next row's diagnosis)
- `FinalDiagnosisv3_Omar`: 18 rows of mojibake + 7 rows with row-shift (case_id 19089-19101),
  plus one case (19145) where the `Section` column leaked into the diagnosis field

These columns appear to have been created as attempted corrections to match Qwen's mojibake
output, but introduced additional errors. They are not used by any evaluation code.

### Fixed: Ophthalmology parse_choice drops letters split by whitespace

**Problem:** When a model returns multi-select answers with an accidental space (e.g.,
`ABCFG H` instead of `ABCFGH`), the `parse_choice` regex `[A-Z]{1,26}` matched only the
first contiguous run, dropping the trailing letter(s).

**Affected data:** 3 predictions in `gpt-5.1-1113` (all 3 runs, row 119: GT=`ABCFGH`,
parsed as `ABCFG`).

**Fix:** Collapse all whitespace before regex matching (`re.sub(r"\s+", "", ...)`).

**Files changed (3):**
- `benchmarks/ophthalmology/hf_bench.py`
- `benchmarks/ophthalmology/openrouter.py`
- `benchmarks/ophthalmology/cot_beams_score.py`
