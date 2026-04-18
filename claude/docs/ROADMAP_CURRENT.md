# Error Analysis Extension — Implementation Plan

Paper: Benchmarking and Adapting On-Device LLMs for Clinical Decision Support
Target: Additive revision, no changes to existing numbers, figures, or tables.

## Goals

1. Characterize error types on the hard tail across model families (5-category taxonomy).
2. Quantify case-level difficulty and split hard cases into *knowledge frontier* (hard for all) vs *capability gap* (recoverable by scale or fine-tuning).
3. Establish best-of-N oracle upper bound on the existing k=3 consensus.

## Scope fences

- Exact-match with `norm_text()` + k=3 majority vote remains the primary scorer. All Table 1–3 numbers unchanged.
- Fig 1, Fig 2, Fig 3 unchanged.
- Abstract unchanged.
- Scorable dataset is the existing 207 graded cases (`FinalDiagnosis` column only).

## Decisions (locked 2026-04-18)

1. **Judge model**: Claude Opus 4.6 via AWS Bedrock (`anthropic.claude-opus-4-6-v1`). Neutral — not in the evaluation set, eliminates self-evaluation bias. Gemini 3.1 Pro as robustness check on the 10% human-validation sample.
2. **Scoring sensitivity**: Dropped. Eurorad is constrained selection from a provided differential list; predictions should be verbatim after `norm_text()`. Taxonomy category (a) quantifies any residual surface-form mismatches. If category (a) > 15% of errors, note in Discussion that exact match conservatively estimates accuracy.
3. **Main-text format**: Table 4 by default; promote to Fig 4 only if space allows.
4. **Taxonomy target**: k=3 majority consensus predictions (not individual runs). Supp Table B reports per-run for completeness.

## Deliverables

- One new Results subsection (~600–800 words)
- One new Methods subsection (~200 words)
- One new main-text figure **or** table (decide in Phase 2)
- Two supplementary tables (Supp Table A: per-run error taxonomy; Supp Table B: single-run vs majority vs oracle)
- One Discussion paragraph, one Limitations sentence
- Scripts committed to `bowang-lab/on-device-LLM` under `analysis/`

---

## Phase 1 — Computation (2–3 days)

### 1.1 Case-level difficulty [DONE]

- Script: `analysis/case_difficulty.py`
- Input: `csvs/final_csvs/Eurorad.csv`, `norm_text()` from `statistics/utils.py`
- For each of 207 cases, compute fraction of models correct (using per-model k=3 majority predictions).
- Cross-tab by model family: proprietary, open-large (DS-R1), on-device base, on-device FT.
- Output: `analysis/outputs/case_difficulty.csv`, ranked case list with per-family correctness columns.
- Acceptance: hard-for-all (bottom decile) and hard-for-small-only (correct by proprietary + FT, wrong by base on-device) case lists saved as separate CSVs.

### 1.2 Best-of-N oracle [DONE]

- Script: `analysis/oracle_bounds.py`
- Input: `csvs/final_csvs/Eurorad.csv`, `norm_text()` from `statistics/utils.py`
- For each model, compute:
  - Single-run accuracy (mean over the 3 runs)
  - k=3 majority accuracy (existing, copy from pipeline)
  - Best-of-3 oracle (correct if any of 3 runs matches)
- Output: `analysis/outputs/oracle_bounds.csv`
- Acceptance: oracle ≥ majority ≥ single-run holds for every model. If not, bug in alignment of runs.

### 1.3 Error taxonomy on the hard tail [DONE]

- Script: `analysis/error_taxonomy.py`
- Input: k=3 majority predictions scored incorrect under exact match (207 cases × 12 models).
- Judge: Claude Opus 4.6 via AWS Bedrock with 5-category rubric:
  - (a) Correct, surface-form mismatch (synonym, eponym, nomenclature variant)
  - (b) Same disease family, wrong specific entity
  - (c) Plausible differential, incorrect
  - (d) Off-topic or hallucinated (not in provided differential list)
  - (e) Empty, refusal, or parse failure
- Prompt includes: clinical history, differential list, model prediction, ground truth. Forces single-category output.
- Human validation: 10% random sample (~50 cases), two annotators (Alif + Omar or Alhusain). Report Cohen's κ.
- Output: `analysis/outputs/error_taxonomy.csv`, per-family aggregated percentages for the stacked bar chart.
- Acceptance: Cohen's κ > 0.7 on validation sample; categories cover ≥ 95% of errors.

---

## Phase 2 — Visualization (1 day) [DONE]

### 2.1 Main-text table [DONE]

Table 4 (`\label{tab:error_taxonomy}`): per-model and per-family error taxonomy with 5 categories (% of errors) + total error count. Grouped by family with subtotals. Placed between error characterization subsection and Methods.

### 2.2 Supplementary artifacts

- Supp Table 4: oracle bounds (single-run vs majority vs oracle per model) — data in `analysis/outputs/oracle_bounds.csv`, referenced in prose
- Supp Table 5: per-run error taxonomy — to be formatted if needed for supplementary PDF
- Optional Supp Figure: deferred to follow-up

---

## Phase 3 — Write-up (2 days) [DONE]

### 3.1 Results — new subsection [DONE]

**Title**: Error characterization across model families
**Placement**: immediately after the fine-tuning subsection, before Methods.

Four paragraphs covering: (1) error taxonomy distribution — 87.2% plausible differential, 5% surface-form, off-topic confined to Qwen 9B/27B; (2) oracle bounds — FT Qwen 35B at 93.2%, Gemma gap=0; (3) hard-for-all — 23 cases of rare entities distributed proportionally across subspecialties; (4) capability-gap mechanism — 84.6% of FT recoveries from gap set, training data exposure explanation.

### 3.2 Methods — new subsection [DONE]

**Title**: Error taxonomy analysis
**Placement**: between "Inference Protocol for Zero-shot Experiments" and "Training Dataset Preparation for Radiological Cases".

Covers: 5-category rubric, Claude Opus 4.6 judge via Bedrock (neutral, external to eval set), prompt structure, case difficulty and capability-gap definitions, oracle computation.

### 3.3 Discussion [DONE]

- One new paragraph after "three key insights", before Limitations: error profile framing (87.2% plausible differential, off-topic as scale effect, capability-gap recovery, shared knowledge frontier).
- One sentence in Limitations: category (a) at 5.0% confirms exact-match conservatively estimates accuracy (included despite <15% threshold because it validates the scoring method).

---

## Phase 4 — Integration (0.5 day)

- Add Results subsection, Methods subsection, Table 4 (or Fig 4), two supp tables to the LaTeX source.
- Verify cross-references: Fig 2c/d still reference correctly; Cases 1–6 now cited in both the anecdotal (existing) and systematic (new) contexts.
- Regenerate supplementary PDF.
- Circulate to co-authors (Omar, Alhusain, Jun, Bo) for review before resubmission.

---

## Risks

- **Rubric drift**: categories blur at edges, especially (b) vs (c). Pre-register rubric with 3 worked examples per category before labeling. Do not iterate mid-run.
- **Scope creep**: "while we're at it" additions (inter-model agreement on errors, per-specialty taxonomy breakdowns, etc.) go to a follow-up, not this revision.

---

## Time estimate

4–6 working days with one contributor. Phase 1 sub-tasks 1.1 and 1.2 can run immediately; 1.3 requires Bedrock API access.
