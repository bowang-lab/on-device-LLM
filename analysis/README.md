# Error Analysis Extension

Scripts for characterizing diagnostic errors across model families on the Eurorad benchmark (207 cases).

## Scripts

| Script | Description | API required |
|---|---|---|
| `case_difficulty.py` | Case-level difficulty: fraction of models correct per case, cross-tabbed by model family. Identifies hard-for-all and capability-gap cases. | No |
| `oracle_bounds.py` | Best-of-N oracle: single-run vs majority-vote vs best-of-3 accuracy per model. Quantifies recoverable performance lost to sampling noise. | No |
| `error_taxonomy.py` | Classifies every incorrect majority-vote prediction into 5 error categories using Claude Opus 4.6 via AWS Bedrock. | Yes (Bedrock) |

## Usage

Run from repo root:

```bash
python analysis/case_difficulty.py
python analysis/oracle_bounds.py
python analysis/error_taxonomy.py            # full run (~$2.50, ~25 min)
python analysis/error_taxonomy.py --dry-run  # count errors without API calls
python analysis/error_taxonomy.py --sample 5 # test on 5 errors
python analysis/error_taxonomy.py --resume   # skip already-classified rows
```

## Outputs

All outputs are written to `analysis/outputs/`.

| File | Source | Description |
|---|---|---|
| `case_difficulty.csv` | 1.1 | 207 rows, one per case. Columns: case_id, section, ground_truth, frac_correct (overall), frac per family, per-model binary. Sorted by difficulty. |
| `hard_for_all.csv` | 1.1 | Bottom-decile cases (correct by <=36% of models). |
| `capability_gap.csv` | 1.1 | Cases correct by proprietary or fine-tuned models but wrong by majority of base on-device models. |
| `oracle_bounds.csv` | 1.2 | Per-model: single-run %, majority %, oracle %, oracle gap, Wilson CIs. |
| `error_taxonomy.csv` | 1.3 | Per-error: case_id, model, family, ground_truth, prediction, category (a-e), justification. |
| `error_taxonomy_by_family.csv` | 1.3 | Aggregated: category counts and percentages per model family. |

## Error Taxonomy Categories

| Category | Label | Description |
|---|---|---|
| (a) | Surface-form mismatch | Clinically correct but differs in wording (synonym, eponym, abbreviation) |
| (b) | Same disease family | Correct organ system or disease group, wrong specific entity |
| (c) | Plausible differential | A differential on the provided list, but incorrect |
| (d) | Off-topic | Not on the differential list, anatomically unrelated, or hallucinated |
| (e) | Empty/refusal | No diagnosis produced, refusal, or parse failure |

## Key Findings (Phase 1)

### Hard-for-all cases (23 cases, ≤36% of models correct)
- Define a shared knowledge frontier of rare histopathological entities: rare tumours with precise subtypes (ganglioneuroblastoma-intermixed, chondroblastic osteosarcoma, RGNT), rare infections (M. kansasii, M. avium), and uncommon syndromes (chorea-acanthocytosis, MASA, Neuro-Behçet's).
- Subspecialty clustering: MSK (5) + neuro (5) = 43% of hard cases, followed by paediatric (4) and abdominal (4).
- No model family dominates — correct answers on 2–4/11 cases are spread across all families.
- Only 3/207 errors in hard cases are category (a) surface mismatches (all from case 19070, clear cell sarcoma). The rest are genuine failures.

### Capability-gap cases (19 cases)
- 84.6% of fine-tuning recoveries come from capability-gap cases — the sharpest mechanistic finding.
- 11/19 (57.9%) overlap with hard-for-all, meaning fine-tuning recovers cases that are difficult even for proprietary models.

### Error taxonomy (358 errors across 11 models)
- 87.2% are category (c) plausible-differential errors across all families.
- Category (a) surface-form mismatches = 5.0% (18 errors), concentrated in 4 GT strings. Confirms exact-match scoring is not substantially underestimating accuracy.
- Category (d) off-topic errors (3.9%) are a scale/family effect, concentrated entirely in Qwen3.5 9B and 27B. All base models ≥31B (including gpt-oss-20b at 3.6B active params) have zero (d) errors.
- Category (e) empty/refusal = 0.3% (1 error, DeepSeek-R1).

### Oracle bounds
- Qwen3.5 35B FT has the highest oracle accuracy (93.2%), largest gap over majority vote.
- Gemma 4 31B oracle gap = 0 (all 3 runs identical on every case).

## Dependencies

- pandas, numpy (standard)
- `statistics/utils.py` for `norm_text()` and `wilson_ci()`
- `boto3` (for error_taxonomy.py only — AWS Bedrock access to Claude Opus 4.6)
