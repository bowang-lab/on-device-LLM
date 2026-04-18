# CSV Directories

**Use `final_csvs/` for all analysis, figures, and statistics.**

| Directory | Description |
|-----------|-------------|
| `final_csvs/` | Corrected data with all fixes applied. All scripts point here. |
| `updated_csvs/` | Raw Gemma 4 additions from collaborators. Eurorad still has unfixed Qwen row-shift and corrupted GT columns. Kept as provenance for Gemma data. |
| Root (`csvs/*.csv`) | Git-tracked originals. No Gemma data, no Qwen fixes. Base files with 1 run per model for NMED. |
| `finetune-eval/` | Per-checkpoint evaluation CSVs for gpt-oss-120b fine-tuning. |

## What's different about `final_csvs/Eurorad.csv`

Built from the git original with three changes:

1. Dropped corrupted GT columns (`FinalDiagnosisv2`, `v2_Alhusain`, `v3_Omar`)
2. Applied Qwen row-shift fix on 9 columns (cases 19041-19087)
3. Merged Gemma 4 columns from `updated_csvs/` (with case_id format normalization)

See `claude/DATA_NOTES.md` for full details on data quality issues and fixes.
