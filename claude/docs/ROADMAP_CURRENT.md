# Error Analysis Extension — Implementation Plan

Paper: Benchmarking and Adapting On-Device LLMs for Clinical Decision Support
Target: Additive revision, no changes to existing numbers, figures, or tables.

## Goals

1. Characterize error types on the hard tail across model families (5-category taxonomy).
2. Quantify case-level difficulty and split hard cases into *knowledge frontier* (hard for all) vs *capability gap* (recoverable by scale or fine-tuning).
3. Establish best-of-N oracle upper bound on the existing k=3 consensus.
4. Pre-empt the scoring-method critique with a sensitivity analysis that leaves headline numbers unchanged.

## Scope fences

- Exact-match with `norm_text()` + k=3 majority vote remains the primary scorer. All Table 1–3 numbers unchanged.
- Fig 1, Fig 2, Fig 3 unchanged.
- Abstract unchanged.
- Scorable dataset is the existing 207 graded cases (`FinalDiagnosis` column only).

## Deliverables

- One new Results subsection (~600–800 words)
- One new Methods subsection (~200 words)
- One new main-text figure **or** table (decide in Phase 2)
- Three supplementary tables
- One Discussion paragraph, one Limitations sentence, one Statistical analysis sentence
- Scripts committed to `bowang-lab/on-device-LLM` under `analysis/`

---

## Phase 1 — Computation (2–3 days)

### 1.1 Case-level difficulty

- Script: `analysis/case_difficulty.py`
- Input: model output CSV, `norm_text()` from `eurorad_stats.py`
- For each of 207 cases, compute fraction of models correct (using per-model k=3 majority predictions).
- Cross-tab by model family: proprietary, open-large (DS-R1), on-device base, on-device FT.
- Output: `outputs/case_difficulty.csv`, ranked case list with per-family correctness columns.
- Acceptance: hard-for-all (bottom decile) and hard-for-small-only (correct by proprietary + FT, wrong by base on-device) case lists saved as separate CSVs.

### 1.2 Best-of-N oracle

- Script: `analysis/oracle_bounds.py`
- For each model, compute:
  - Single-run accuracy (mean over the 3 runs)
  - k=3 majority accuracy (existing, copy from pipeline)
  - Best-of-3 oracle (correct if any of 3 runs matches)
- Output: `outputs/oracle_bounds.csv`
- Acceptance: oracle ≥ majority ≥ single-run holds for every model. If not, bug in alignment of runs.

### 1.3 Scoring sensitivity

- Script: `analysis/scoring_sensitivity.py`
- Re-score the same k=3 majority predictions under:
  - Exact match (existing `norm_text()`)
  - Fuzzy: `rapidfuzz.token_set_ratio ≥ 85`
  - LLM-judge: GPT-5.1 with 3-way rubric (correct / accepted synonym / incorrect)
- Output: per-model accuracy under each scorer, Spearman ρ of model rankings across scorers.
- Acceptance: ρ > 0.9 across scorers. If not, flag in decisions below.

### 1.4 Error taxonomy on the hard tail

- Script: `analysis/error_taxonomy.py`
- Input: k=3 majority predictions scored incorrect under exact match (207 cases × ~19 models).
- Judge: GPT-5.1 with 5-category rubric:
  - (a) Correct, surface-form mismatch (synonym, eponym, nomenclature variant)
  - (b) Same disease family, wrong specific entity
  - (c) Plausible differential, incorrect
  - (d) Off-topic or hallucinated (not in provided differential list)
  - (e) Empty, refusal, or parse failure
- Prompt includes: clinical history, differential list, model prediction, ground truth. Forces single-category output.
- Human validation: 10% random sample (~50 cases), two annotators (Alif + Omar or Alhusain). Report Cohen's κ.
- Output: `outputs/error_taxonomy.csv`, per-family aggregated percentages for the stacked bar chart.
- Acceptance: Cohen's κ > 0.7 on validation sample; categories cover ≥ 95% of errors.

---

## Phase 2 — Visualization (1 day)

### 2.1 Main-text figure or table

Decision: default to **Table 4** (lower reviewer risk, preserves figure budget). Move to figure only if journal allows Fig 4.

- **Table 4 option**: rows = model families, columns = 5 error categories (% of errors each), plus total error rate. Short caption.
- **Fig 4 option** (if used):
  - Panel (a): case difficulty histogram, cases coloured by hard-for-all vs recoverable-by-scale
  - Panel (b): stacked error-category bars, one per model family

### 2.2 Supplementary artifacts

- Supp Table A: scoring sensitivity (3 scorers × all models, plus Spearman ρ)
- Supp Table B: full per-run error taxonomy (not collapsed to families)
- Supp Table C: single-run vs majority vs oracle per model
- Optional Supp Figure: per-subspecialty error-type distribution

---

## Phase 3 — Write-up (2 days)

### 3.1 Results — new subsection

**Title**: Error characterization across model families
**Placement**: immediately after the fine-tuning subsection, before Discussion.

- **Paragraph 1**: case-level difficulty + best-of-N oracle. Framing: gap between majority and oracle separates sampling noise from knowledge limits. Identify the knowledge-frontier subset.
- **Paragraph 2**: error taxonomy across families. Quantitative statement of the form "X% of Qwen3.5-35B errors fall in categories (a) and (b), i.e. the diagnosis is in the right neighbourhood." Link Cases 1–6 as qualitative instances of these categories.
- **Paragraph 3**: what fine-tuning preferentially corrects. Category-level before/after for gpt-oss-20b and Qwen3.5-35B. This is the paragraph that ties the new analysis to the paper's thesis.

### 3.2 Methods — new subsection

**Title**: Error taxonomy analysis
**Placement**: between "Inference Protocol for Zero-shot Experiments" and "Training Dataset Preparation for Radiological Cases".

Covers: 5-category rubric, judge model (GPT-5.1 for consistency with leakage screen), prompt structure, human validation protocol, agreement metric (Cohen's κ), fuzzy threshold (0.85, rapidfuzz token set ratio).

### 3.3 Discussion

- One new paragraph after the "three key insights" paragraph, before Limitations.
  - Frame: error characterization reveals a specific failure profile; residual errors in the strongest on-device models are mostly near-miss rather than off-list; fine-tuning preferentially reduces category (b) and (c); hard-for-all cases map to the shared knowledge frontier.
- One sentence in Limitations, after the retrospective-data sentence:
  - "Exact-match scoring conservatively estimates diagnostic capability; a sensitivity analysis under fuzzy and LLM-judge scoring confirmed that model rankings are preserved (Supplementary Table A)."

### 3.4 Statistical analysis

- Add one sentence noting the scoring sensitivity check and ranking stability (Spearman ρ, Supp Table A).

---

## Phase 4 — Integration (0.5 day)

- Add Results subsection, Methods subsection, Table 4 (or Fig 4), three supp tables to the LaTeX source.
- Verify cross-references: Fig 2c/d still reference correctly; Cases 1–6 now cited in both the anecdotal (existing) and systematic (new) contexts.
- Regenerate supplementary PDF.
- Circulate to co-authors (Omar, Alhusain, Jun, Bo) for review before resubmission.

---

## Decisions to lock before starting

1. **Judge model**: GPT-5.1 alone, with Gemini 3.1 Pro as a robustness check on the 10% validation sample.
2. **Fuzzy threshold**: rapidfuzz `token_set_ratio ≥ 85`. Validate on 20 hand-picked cases (e.g. "Lymphoma" vs "Primary cardiac lymphoma") before committing.
3. **Main-text format**: Table 4 by default; promote to Fig 4 only if space allows.
4. **Fallback if rankings not stable under fuzzy/judge**: this would be a genuine finding that changes the headline framing. Unlikely given clustering at the top, but the response is to report both scorers side by side in the abstract rather than pick one. Predefine this now.
5. **Taxonomy target**: k=3 majority consensus predictions (not individual runs). Supp Table B reports per-run for completeness.

---

## Risks

- **Judge bias**: GPT-5.1 judging errors that include its own outputs. Human validation sample catches systematic bias; if detected, rerun taxonomy with Gemini 3.1 Pro as judge.
- **Rubric drift**: categories blur at edges, especially (b) vs (c). Pre-register rubric with 3 worked examples per category before labeling. Do not iterate mid-run.
- **Scope creep**: "while we're at it" additions (inter-model agreement on errors, per-specialty taxonomy breakdowns, etc.) go to a follow-up, not this revision.

---

## Time estimate

5–7 working days with one contributor. Phase 1 sub-tasks are parallelizable across people if available.