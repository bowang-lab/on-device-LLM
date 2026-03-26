# Results

CSV files extracted from `Model_output.xlsx`. Each file corresponds to one sheet.

## Formulas & Statistical Methods

The original spreadsheet used **25+ custom Google Sheets (Apps Script) functions** to compute statistics. These are preserved in [`formulas.md`](formulas.md), which contains:

- A summary table of all custom functions (with usage counts and descriptions)
- Every formula listed by sheet and cell reference

Key statistical methods used:
- **Accuracy**: Exact-match and self-consistency (majority vote) with Wilson score CIs
- **Likert scoring**: MAE, median, IQR, and 95% CIs on Likert-scale human evaluation scores
- **Pairwise model comparison**: McNemar's test (categorical), Wilcoxon signed-rank test (ordinal)
- **Inter-rater / cross-run agreement**: Cohen's kappa, weighted kappa, Fleiss' kappa, ICC

## Summary & Paper Tables

| File | Rows | Description |
|------|------|-------------|
| `summary.csv` | ~1063 | High-level performance summary across all models. Columns: Model, Admin, Eurorad (Exact Match), Ophthalmology (EM), NMED Diagnosis (MAE), NMED Treatment (MAE), with run counts. |
| `paper_tables.csv` | ~216 | Formatted results for the paper: model accuracy with 95% confidence intervals across all four benchmarks. |
| `final_paper_tables.csv` | ~230 | Final paper-ready tables using self-consistency (majority vote) aggregation across runs. |
| `Alhusain_NMED.csv` | ~13 | Compact summary table of model performance (accuracy + CI) across all four benchmarks, curated by Alhusain. |

## Eurorad (Radiology Diagnosis)

| File | Rows | Description |
|------|------|-------------|
| `Eurorad.csv` | ~1180 | Full Eurorad benchmark results. Each row is a radiology case with case_id, body section, clinical description (original and post-processed), differential diagnosis list, final diagnosis, disease leak flag, and model predictions across 3 runs for GPT-5, GPT-5.1, GPT-5.2, GPT-5-mini, o4-mini, Gemini-3.1-pro, DeepSeek-R1, oss-20b (L/M/H), oss-120b (L/M/H), Qwen-3.5 (9B/27B/35B), and fine-tuned variants. |
| `Copy_of_Eurorad.csv` | ~998 | Earlier version of Eurorad results with a subset of models (GPT-5, o4-mini, DeepSeek-R1, oss-20b, oss-120b). |
| `Alif_Eurorad.csv` | ~1000 | Eurorad accuracy broken down by anatomical category (e.g., Musculoskeletal, Neuro, Chest) with per-model accuracy and CIs. |
| `Outlier_cases_for_Eurorad.csv` | ~5 | Eurorad cases identified as outliers — edge cases where models consistently struggled. |
| `finetune-Eurorad.csv` | ~1000 | Eurorad results for fine-tuned and beam-search variants of oss-20b (5/9/13 beams, MoE-Linear fine-tuned). |
| `Eurorad_Train_Sections.csv` | ~1892 | Eurorad training set with full case metadata: case_id, title, publication date, URL, DOI, body section, and all text sections (clinical history, imaging findings, discussion, differential diagnosis list, final diagnosis, outcome, teaching points, references). |

## Ophthalmology

| File | Rows | Description |
|------|------|-------------|
| `Ophthalmology.csv` | ~1004 | Full ophthalmology benchmark results. Each row is a multiple-choice question with topic, type (Diagnosis/Treatment), question text, ground truth answer, and model predictions across 3 runs for all models. |
| `Omar_Opthalmology.csv` | ~105 | Ophthalmology accuracy by topic (e.g., Glaucoma, Retina) and type, with per-model accuracy and CIs. Curated by Omar. |
| `OmarOpthalmologyUpdated.csv` | ~80 | Updated version of Omar's ophthalmology summary (may contain formatting updates). |

## NMED Diagnosis

| File | Rows | Description |
|------|------|-------------|
| `NMED_Diagnosis.csv` | ~1009 | NMED diagnosis benchmark — single-run results. Each row: clinical specialty, ground-truth disease, disease frequency, model name, disease description prompt + model output, human evaluation score, and model scores for GPT-5, o4-mini, DeepSeek, oss-20b (L/M/H), oss-120b (L/M/H). |
| `NMED_Diagnosis_3x.csv` | ~1008 | NMED diagnosis with 3 runs per model. Same structure but with triplicate columns (v1/v2/v3) for all models including GPT-5.1, GPT-5.2, Gemini, Qwen variants. |
| `Copy_of_NMED_Diagnosis.csv` | ~1000 | Earlier version of NMED Diagnosis with additional columns for DeepSeek runs and oss-20b beam-search variants. |
| `NMED_Diagnosis_beams.csv` | ~731 | NMED diagnosis results comparing different beam-search configurations (5/9/11 beams) for oss-20b. |

## NMED Treatment

| File | Rows | Description |
|------|------|-------------|
| `NMED_Treatment.csv` | ~1000 | NMED treatment benchmark — single-run results. Same structure as NMED Diagnosis but for treatment recommendation tasks. |
| `NMED_Treatment_3x.csv` | ~1003 | NMED treatment with 3 runs per model across all models. |
| `Copy_of_NMED_Treatment.csv` | ~1000 | Earlier version of NMED Treatment with additional DeepSeek runs and beam-search columns. |
| `NMED_Treatment_beams.csv` | ~607 | NMED treatment results comparing beam-search configurations (5/9/11 beams) for oss-20b. |

## NMED Analysis

| File | Rows | Description |
|------|------|-------------|
| `nmed_specialty.csv` | ~79 | NMED performance broken down by clinical specialty, with IQR values across models (GPT-5.1, Gemini, GPT-5-mini, DeepSeek, gpt-oss, Qwen 9B/27B/35B). |
| `nmed_frequency.csv` | ~32 | NMED performance stratified by disease frequency (All, Rare, Less Frequent, Frequent) for diagnosis and treatment tasks. |

## Model Outputs & Prompt Engineering

| File | Rows | Description |
|------|------|-------------|
| `Model_output.csv` | ~1009 | Raw model outputs across all tasks (Eurorad, NMED, Ophthalmology). Columns: Task, system prompt, case ID, ground truth, and raw outputs from GPT-5.2, GPT-5.1, GPT-5-mini, Gemini 3.1 Pro. |
| `oss20b_med_promptim_v1.csv` | ~1000 | Eurorad results for oss-20b (medium setting) with prompt optimization v1. Includes raw model answer, parsed answer, match type, correctness flag, and answer options. |
| `oss20b_high_promptim_v1.csv` | ~1000 | Eurorad results for oss-20b (high setting) with prompt optimization v1. Same structure as medium variant. |
