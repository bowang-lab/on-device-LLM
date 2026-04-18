# Roadmap

Known issues, potential improvements, and planned work.

---

## Data Quality

### Clean up v2/v3 ground truth columns in Eurorad
**Status:** Needs decision
**Context:** `FinalDiagnosisv2`, `FinalDiagnosisv2_Alhusain`, and `FinalDiagnosisv3_Omar`
in `csvs/Eurorad.csv` and `csvs/Model_output.xlsx` have mojibake + row-shift bugs
(see CHANGELOG 2026-04-02). Options:
1. Drop the columns entirely (they aren't used by eval code)
2. Fix them: apply CP1252 repair and correct the row shifts
3. Keep as-is with documentation (current state)

### Audit DifferentialDiagnosisList for answer leakage
The `DiseaseLeak` column exists but its coverage and methodology haven't been reviewed.

---

## Evaluation Pipeline

### Consolidate text normalization into a shared module
Nine files define their own `norm_text()` / `normalize_text()` with slight variations
(some strip special chars, some don't; some remove quotes, some don't). A single
`utils/text.py` module would prevent drift and make future fixes one-line changes.

### Recompute Eurorad statistics with corrected Qwen scores
The mojibake fix changes Qwen accuracy by 7-9%. The stats in `csvs/eurorad_stats.csv`
and `csvs/paper_tables.csv` need regeneration via `compute_eurorad_stats.py`.

### Investigate Qwen tokenizer mojibake at the source
The CP1252 recovery in `norm_text()` is a workaround. The root cause is in the Qwen 3.5
inference pipeline -- the tokenizer or decoding step is emitting corrupted bytes. Fixing
this upstream would eliminate the need for post-hoc repair.

---

## Paper

### Update tables with corrected Qwen results
After recomputing stats, the LaTeX tables in `paper/main.tex` and `paper/supp.tex` will
need updated numbers for all Qwen 3.5 variants.

---

## GEPA Prompt Optimization

### Run GEPA optimize_anything on gpt-oss-120b Eurorad
**Status:** Script ready (`gepa/gepa_oa_eurorad.py`), awaiting execution.
**Context:** Automated system prompt optimization using Bedrock Claude Sonnet 4 as
reflection LM and gpt-oss-120b (H) as task model. Budget: 300 metric calls. Target:
improve from 83.1% baseline toward 85-87%. See `claude/docs/GEPA_PROMPT_OPTIMIZATION.md`.

### Evaluate optimized prompt with self-consistency (k=3)
**Status:** Blocked on above.
**Context:** After GEPA finds the best prompt, re-evaluate with k=3 runs and majority
vote to produce paper-ready numbers with Wilson CIs. Use existing `hf_bench.py` with
`--system` flag pointing to the optimized prompt.

### Consider GEPA on gpt-oss-20b-FT
**Status:** Future.
**Context:** Test whether prompt optimization stacks with fine-tuning. If gpt-oss-20b-FT
(86.5%) gains further from an optimized prompt, that strengthens the paper's adaptation
story.

---

## Fine-tuning

### Evaluate whether fine-tuned Qwen models need retraining
If the Qwen mojibake also affected training data preparation, fine-tuned model quality
may have been impacted. Check `finetune/` data pipelines for the same encoding issue.
