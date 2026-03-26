# Formulas Extracted from Model_output.xlsx

These are the Google Sheets custom functions (Apps Script) and built-in formulas
used to compute the statistics in this workbook. The CSVs contain only the
resolved values; this file preserves the computation logic.

## Custom Statistical Functions

| Function | Uses | Description |
|----------|------|-------------|
| `CALC_ACCURACY_UPDATED` | 1135 | Calculates accuracy (exact match) for model predictions against ground truth |
| `REGEXEXTRACT` | 418 | Extracts substrings via regex (Google Sheets built-in) |
| `LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR` | 260 | Computes mean consensus score with median and IQR for Likert-scale evaluations |
| `LIKERT_CONSENSUS_MAE_CI` | 222 | Computes MAE with 95% confidence intervals for Likert-scale scores |
| `CALC_ACCURACY` | 140 | Earlier version of accuracy calculation |
| `MAE` | 139 | Mean Absolute Error between model scores and human evaluation scores |
| `LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR` | 108 | Signed consensus median with IQR for Likert data |
| `MODEL_ACCURACY` | 91 | Model-level accuracy aggregation |
| `MCNEMAR_VOTE_TEST` | 87 | McNemar's test on majority-vote predictions for pairwise model comparison |
| `COMBINE_DIAG_MANAG` | 85 | Combines diagnosis and management scores |
| `WILCOXON_CONSENSUS_TEST` | 73 | Wilcoxon signed-rank test on consensus scores for pairwise model comparison |
| `COHEN_KAPPA_CONSENSUS` | 72 | Cohen's kappa inter-rater agreement on consensus predictions |
| `WEIGHTED_KAPPA_CONSENSUS` | 72 | Weighted kappa for ordinal agreement on consensus predictions |
| `MAE_IQR_TEXT` | 54 | Formats MAE with IQR as text for paper tables |
| `MEDAE_IQR_TEXT` | 54 | Formats Median AE with IQR as text |
| `MEAN_SCORE_CI_TEXT` | 54 | Formats mean score with CI as text |
| `WILCOXON_SIGNED_RANK_TEXT` | 54 | Formats Wilcoxon test results as text |
| `ACCURACY` | 48 | Basic accuracy calculation |
| `AVERAGE` | 45 | Built-in Excel/Sheets average |
| `MAE_CI_TEXT` | 42 | Formats MAE with CI as text |
| `MCQ_CI_TEXT` | 29 | Formats MCQ accuracy with CI as text |
| `SC_ACCURACY_CI` | 19 | Self-consistency accuracy with confidence intervals |
| `FLEISS_KAPPA_ROBUST` | 18 | Fleiss' kappa for multi-rater agreement across runs |
| `ICC_RELIABILITY` | 18 | Intraclass Correlation Coefficient for score reliability across runs |
| `SC_ACCURACY_WILSON` | 10 | Self-consistency accuracy with Wilson score CI |
| `MCNEMAR_TEST` | 2 | Standard McNemar's test (single-run) |
| `WILCOXON_ERROR_TEST` | 2 | Wilcoxon test on error distributions |

## All Formulas by Sheet

### summary (20 formulas)

| Cell | Formula |
|------|--------|
| D2 | `=AVERAGE(D16:D18)` |
| F2 | `=AVERAGE(F16:F18)` |
| D4 | `=AVERAGE(D19:D21)` |
| F4 | `=AVERAGE(F19:F21)` |
| D5 | `=AVERAGE(D22:D24)` |
| F5 | `=AVERAGE(F22:F24)` |
| D6 | `=AVERAGE(D25:D27)` |
| F6 | `=AVERAGE(F25:F27)` |
| D7 | `=AVERAGE(D28:D30)` |
| F7 | `=AVERAGE(F28:F30)` |
| D8 | `=AVERAGE(D31:D33)` |
| F8 | `=AVERAGE(F31:F33)` |
| D9 | `=AVERAGE(D34:D36)` |
| F9 | `=AVERAGE(F34:F36)` |
| D10 | `=AVERAGE(D37:D39)` |
| F10 | `=AVERAGE(F37:F39)` |
| D11 | `=AVERAGE(D40:D42)` |
| F11 | `=AVERAGE(F40:F42)` |
| D12 | `=AVERAGE(D43:D45)` |
| F12 | `=AVERAGE(F43:F45)` |

### paper tables (324 formulas)

| Cell | Formula |
|------|--------|
| B19 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| B20 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| C20 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| B21 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| C21 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| D21 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| B22 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| C22 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| D22 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| E22 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| B23 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| C23 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| D23 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| E23 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| F23 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| B24 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| C24 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| D24 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| E24 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| F24 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| G24 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| B25 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| C25 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| D25 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| E25 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| F25 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| G25 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| H25 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| B26 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| C26 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| D26 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| E26 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| F26 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| G26 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| H26 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| I26 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| B34 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| B35 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| C35 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| B36 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| C36 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| D36 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| B37 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| C37 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| D37 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| E37 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| B38 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| C38 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| D38 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| E38 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| F38 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| B39 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| C39 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| D39 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| E39 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| F39 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| G39 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| B40 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| C40 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| D40 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| E40 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| F40 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| G40 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| H40 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| B41 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| C41 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| D41 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| E41 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| F41 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| G41 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| H41 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| I41 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| B83 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$596, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| B84 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$596, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| C84 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| B85 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$596, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| C85 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| D85 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| B86 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$596, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| C86 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| D86 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| E86 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| B87 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$596, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| C87 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| D87 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| E87 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| F87 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| B88 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$596, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| C88 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$596, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| D88 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| E88 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| F88 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| G88 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| B89 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| C89 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| D89 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| E89 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| F89 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| G89 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| H89 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| B90 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| C90 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| D90 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| E90 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| F90 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| G90 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| H90 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| I90 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Treatment 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Treatment 3x'!$1:$1, 0)),   FILTER('NMED Treatment 3x'!$2:$721, REGEXMAT...` |
| B109 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| B110 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| C110 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| B111 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| C111 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| D111 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| B112 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| C112 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| D112 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| E112 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| B113 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| C113 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| D113 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| E113 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| F113 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| B114 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| C114 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| D114 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| E114 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| F114 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| G114 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| B115 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| C115 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| D115 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| E115 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| F115 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| G115 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| H115 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| B116 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| C116 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| D116 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| E116 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| F116 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| G116 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| H116 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| I116 | `=IFERROR(__xludf.DUMMYFUNCTION("WILCOXON_CONSENSUS_TEST(   INDEX('NMED Diagnosis 3x'!$2:$721, , MATCH(""HumanEvalScore"", 'NMED Diagnosis 3x'!$1:$1, 0)),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| B122 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)gpt-5"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)o4.*mini"")) )"),"...` |
| B123 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)gpt-5"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)deepseek"")) )"),"...` |
| C123 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)o4.*mini"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)deepseek"")) )"...` |
| B124 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)gpt-5"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(L\)...` |
| C124 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)o4.*mini"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(...` |
| D124 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)deepseek"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(...` |
| B125 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)gpt-5"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(M\)...` |
| C125 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)o4.*mini"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(...` |
| D125 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)deepseek"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(...` |
| E125 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[-...` |
| B126 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)gpt-5"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(H\)...` |
| C126 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)o4.*mini"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(...` |
| D126 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)deepseek"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(...` |
| E126 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[-...` |
| F126 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[-...` |
| B127 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)gpt-5"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?120b.*\(L\...` |
| C127 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)o4.*mini"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?120b.*\...` |
| D127 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)deepseek"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?120b.*\...` |
| E127 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[-...` |
| F127 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[-...` |
| G127 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[-...` |
| B128 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)gpt-5"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?120b.*\(M\...` |
| C128 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)o4.*mini"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?120b.*\...` |
| D128 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)deepseek"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?120b.*\...` |
| E128 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[-...` |
| F128 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[-...` |
| G128 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[-...` |
| H128 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?120b.*\(L\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[...` |
| B129 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)gpt-5"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?120b.*\(H\...` |
| C129 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)o4.*mini"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?120b.*\...` |
| D129 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)deepseek"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?120b.*\...` |
| E129 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[-...` |
| F129 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[-...` |
| G129 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[-...` |
| H129 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?120b.*\(L\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[...` |
| I129 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[- ]?120b.*\(M\)"")),   FILTER(Eurorad!$2:$208, REGEXMATCH(Eurorad!$1:$1, ""(?i)oss[...` |
| B135 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)gpt5"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, "...` |
| B136 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)gpt5"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, "...` |
| C136 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)o4.*mini"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$...` |
| B137 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)gpt5"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, "...` |
| C137 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)o4.*mini"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$...` |
| D137 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)ds"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(...` |
| B138 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)gpt5"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, "...` |
| C138 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)o4.*mini"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$...` |
| D138 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)ds"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(...` |
| E138 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalm...` |
| B139 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)gpt5"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, "...` |
| C139 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)o4.*mini"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$...` |
| D139 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)ds"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(...` |
| E139 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalm...` |
| F139 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalm...` |
| B140 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)gpt5"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, "...` |
| C140 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)o4.*mini"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$...` |
| D140 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)ds"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(...` |
| E140 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalm...` |
| F140 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalm...` |
| G140 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalm...` |
| B141 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)gpt5"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, "...` |
| C141 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)o4.*mini"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$...` |
| D141 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)ds"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(...` |
| E141 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalm...` |
| F141 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalm...` |
| G141 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalm...` |
| H141 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?120b.*\(L\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthal...` |
| B142 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)gpt5"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, "...` |
| C142 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)o4.*mini"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$...` |
| D142 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)ds"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(...` |
| E142 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalm...` |
| F142 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalm...` |
| G142 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalm...` |
| H142 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?120b.*\(L\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthal...` |
| I142 | `=IFERROR(__xludf.DUMMYFUNCTION("COHEN_KAPPA_CONSENSUS(   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthalmology!$1:$1, ""(?i)oss[- ]?120b.*\(M\)"")),   FILTER(Ophthalmology!$2:$131, REGEXMATCH(Ophthal...` |
| B148 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('...` |
| B149 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('...` |
| C149 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| B150 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('...` |
| C150 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| D150 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)deepseek"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| B151 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('...` |
| C151 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| D151 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)deepseek"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| E151 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER('NMED Treatment 3x'!$2:$596...` |
| B152 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('...` |
| C152 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| D152 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)deepseek"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| E152 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER('NMED Treatment 3x'!$2:$596...` |
| F152 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER('NMED Treatment 3x'!$2:$596...` |
| B153 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('...` |
| C153 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| D153 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)deepseek"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| E153 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER('NMED Treatment 3x'!$2:$596...` |
| F153 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER('NMED Treatment 3x'!$2:$596...` |
| G153 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")),   FILTER('NMED Treatment 3x'!$2:$596...` |
| B154 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('...` |
| C154 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| D154 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)deepseek"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| E154 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER('NMED Treatment 3x'!$2:$596...` |
| F154 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER('NMED Treatment 3x'!$2:$596...` |
| G154 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")),   FILTER('NMED Treatment 3x'!$2:$596...` |
| H154 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?120b.*\(L\)"")),   FILTER('NMED Treatment 3x'!$2:$59...` |
| B155 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('...` |
| C155 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| D155 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)deepseek"")),   FILTER('NMED Treatment 3x'!$2:$596, REGEXMAT...` |
| E155 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER('NMED Treatment 3x'!$2:$596...` |
| F155 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER('NMED Treatment 3x'!$2:$596...` |
| G155 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")),   FILTER('NMED Treatment 3x'!$2:$596...` |
| H155 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?120b.*\(L\)"")),   FILTER('NMED Treatment 3x'!$2:$59...` |
| I155 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?120b.*\(M\)"")),   FILTER('NMED Treatment 3x'!$2:$59...` |
| B161 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('...` |
| B162 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('...` |
| C162 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| B163 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('...` |
| C163 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| D163 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)deepseek"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| B164 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('...` |
| C164 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| D164 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)deepseek"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| E164 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER('NMED Diagnosis 3x'!$2:$721...` |
| B165 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('...` |
| C165 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| D165 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)deepseek"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| E165 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER('NMED Diagnosis 3x'!$2:$721...` |
| F165 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER('NMED Diagnosis 3x'!$2:$721...` |
| B166 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('...` |
| C166 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| D166 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)deepseek"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| E166 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER('NMED Diagnosis 3x'!$2:$721...` |
| F166 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER('NMED Diagnosis 3x'!$2:$721...` |
| G166 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")),   FILTER('NMED Diagnosis 3x'!$2:$721...` |
| B167 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('...` |
| C167 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| D167 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)deepseek"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| E167 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER('NMED Diagnosis 3x'!$2:$721...` |
| F167 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER('NMED Diagnosis 3x'!$2:$721...` |
| G167 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")),   FILTER('NMED Diagnosis 3x'!$2:$721...` |
| H167 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?120b.*\(L\)"")),   FILTER('NMED Diagnosis 3x'!$2:$72...` |
| B168 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)gpt5"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('...` |
| C168 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)o4.*mini"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| D168 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)deepseek"")),   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMAT...` |
| E168 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")),   FILTER('NMED Diagnosis 3x'!$2:$721...` |
| F168 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")),   FILTER('NMED Diagnosis 3x'!$2:$721...` |
| G168 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")),   FILTER('NMED Diagnosis 3x'!$2:$721...` |
| H168 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?120b.*\(L\)"")),   FILTER('NMED Diagnosis 3x'!$2:$72...` |
| I168 | `=IFERROR(__xludf.DUMMYFUNCTION("WEIGHTED_KAPPA_CONSENSUS(   FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?120b.*\(M\)"")),   FILTER('NMED Diagnosis 3x'!$2:$72...` |
| B172 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH(TRANSPOSE(Eurorad!$1:$1), ""^gpt-5"")     )   ) ) "),"Loading...")` |
| B173 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH(TRANSPOSE(Eurorad!$1:$1), ""^o4-mini"")     )   ) ) "),"Loading...")` |
| B174 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH(TRANSPOSE(Eurorad!$1:$1), ""^deepseek"")     )   ) ) "),"Loading...")` |
| B175 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH(TRANSPOSE(Eurorad!$1:$1), ""^oss-20b \(L\)"")     )   ) ) "),"Loading.....` |
| B176 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH(TRANSPOSE(Eurorad!$1:$1), ""^oss-20b \(M\)"")     )   ) ) "),"Loading.....` |
| B177 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH(TRANSPOSE(Eurorad!$1:$1), ""^oss-20b \(H\)"")     )   ) ) "),"Loading.....` |
| B178 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH(TRANSPOSE(Eurorad!$1:$1), ""^oss-120b \(L\)"")     )   ) ) "),"Loading....` |
| B179 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH(TRANSPOSE(Eurorad!$1:$1), ""^oss-120b \(M\)"")     )   ) ) "),"Loading....` |
| B180 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH(TRANSPOSE(Eurorad!$1:$1), ""^oss-120b \(H\)"")     )   ) ) "),"Loading....` |
| B184 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGEXMATCH(TRANSPOSE(Ophthalmology!$1:$1), ""^gpt5"")     )   ), TRUE ) "),"...` |
| B185 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGEXMATCH(TRANSPOSE(Ophthalmology!$1:$1), ""^o4-mini"")     )   ), TRUE ) "...` |
| B186 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGEXMATCH(TRANSPOSE(Ophthalmology!$1:$1), ""^ds-r1"")     )   ), TRUE ) "),...` |
| B187 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGEXMATCH(TRANSPOSE(Ophthalmology!$1:$1), ""^oss20b \(L\)"")     )   ), TRU...` |
| B188 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGEXMATCH(TRANSPOSE(Ophthalmology!$1:$1), ""^oss20b \(M\)"")     )   ), TRU...` |
| B189 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGEXMATCH(TRANSPOSE(Ophthalmology!$1:$1), ""^oss20b \(H\)"")     )   ), TRU...` |
| B190 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGEXMATCH(TRANSPOSE(Ophthalmology!$1:$1), ""^oss120b \(L\)"")     )   ), TR...` |
| B191 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGEXMATCH(TRANSPOSE(Ophthalmology!$1:$1), ""^oss120b \(M\)"")     )   ), TR...` |
| B192 | `=IFERROR(__xludf.DUMMYFUNCTION("FLEISS_KAPPA_ROBUST(   TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGEXMATCH(TRANSPOSE(Ophthalmology!$1:$1), ""^oss120b \(H\)"")     )   ), TR...` |
| B196 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)gpt5"")))"),"Loading...")` |
| B197 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)o4-mini"")))"),"Loading...")` |
| B198 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)deepseek"")))"),"Loading...")` |
| B199 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")))"),"Loading...")` |
| B200 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")))"),"Loading...")` |
| B201 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")))"),"Loading...")` |
| B202 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?120b.*\(L\)"")))"),"Loading...")` |
| B203 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?120b.*\(M\)"")))"),"Loading...")` |
| B204 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Treatment 3x'!$2:$596, REGEXMATCH('NMED Treatment 3x'!$1:$1, ""(?i)oss[- ]?120b.*\(H\)"")))"),"Loading...")` |
| B208 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)gpt5"")))"),"Loading...")` |
| B209 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)o4-mini"")))"),"Loading...")` |
| B210 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)deepseek"")))"),"Loading...")` |
| B211 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(L\)"")))"),"Loading...")` |
| B212 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(M\)"")))"),"Loading...")` |
| B213 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?20b.*\(H\)"")))"),"Loading...")` |
| B214 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?120b.*\(L\)"")))"),"Loading...")` |
| B215 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?120b.*\(M\)"")))"),"Loading...")` |
| B216 | `=IFERROR(__xludf.DUMMYFUNCTION("ICC_RELIABILITY(FILTER('NMED Diagnosis 3x'!$2:$721, REGEXMATCH('NMED Diagnosis 3x'!$1:$1, ""(?i)oss[- ]?120b.*\(H\)"")))"),"Loading...")` |

### nmed_specialty (372 formulas)

| Cell | Formula |
|------|--------|
| B5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[gpt-5.1-1113-m1],   Table1...` |
| C5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[gpt-5.1-1113-m1],  ...` |
| D5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[gpt-5.1-1113-m1],   Table1_...` |
| E5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[gpt-5.1-1113-m1],   Table1...` |
| F5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[gpt-5.1-1113-m1],   Table1_3[...` |
| G5 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[gpt-5.1-1113-m1],Table1_3[gpt-5.1-1113-m2], Table1_3[gpt-5.1-1113-m3])` |
| J5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[gpt-5.1-1113-m1],   Table2...` |
| K5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[gpt-5.1-1113-m1],  ...` |
| L5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[gpt-5.1-1113-m1],   Table2_...` |
| M5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[gpt-5.1-1113-m1],   Table2...` |
| N5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[gpt-5.1-1113-m1],   Table2_3[...` |
| O5 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[gpt-5.1-1113-m1],   Table2_3[gpt-5.1-1113-m2],   Table2_3[gpt-5.1-1113-m3] )` |
| B6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[gemini-3.1-pro-m1],   Tabl...` |
| C6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[gemini-3.1-pro-m1],...` |
| D6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[gemini-3.1-pro-m1],   Table...` |
| E6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[gemini-3.1-pro-m1],   Tabl...` |
| F6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[gemini-3.1-pro-m1],   Table1_...` |
| G6 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[gemini-3.1-pro-m1],Table1_3[gemini-3.1-pro-m2], Table1_3[gemini-3.1-pro-m3])` |
| J6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[gemini-3.1-pro-m1],   Tabl...` |
| K6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[gemini-3.1-pro-m1],...` |
| L6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[gemini-3.1-pro-m1],   Table...` |
| M6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[gemini-3.1-pro-m1],   Tabl...` |
| N6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[gemini-3.1-pro-m1],   Table2_...` |
| O6 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[gemini-3.1-pro-m1],   Table2_3[gemini-3.1-pro-m2],   Table2_3[gemini-3.1-pro-m3] )` |
| B7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[gpt-5-mini-0807-m1],   Tab...` |
| C7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[gpt-5-mini-0807-m1]...` |
| D7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[gpt-5-mini-0807-m1],   Tabl...` |
| E7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[gpt-5-mini-0807-m1],   Tab...` |
| F7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[gpt-5-mini-0807-m1],   Table1...` |
| G7 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[gpt-5-mini-0807-m1],Table1_3[gpt-5-mini-0807-m2], Table1_3[gpt-5-mini-0807-m3])` |
| J7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[gpt-5-mini-0807-m1],   Tab...` |
| K7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[gpt-5-mini-0807-m1]...` |
| L7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[gpt-5-mini-0807-m1],   Tabl...` |
| M7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[gpt-5-mini-0807-m1],   Tab...` |
| N7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[gpt-5-mini-0807-m1],   Table2...` |
| O7 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[gpt-5-mini-0807-m1],   Table2_3[gpt-5-mini-0807-m2],   Table2_3[gpt-5-mini-0807-m3] )` |
| B8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Ta...` |
| C8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[deepseek-r1-0528-v1...` |
| D8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Tab...` |
| E8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Ta...` |
| F8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table...` |
| G8 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[deepseek-r1-0528-v1],Table1_3[deepseek-r1-0528-v2], Table1_3[deepseek-r1-0528-v3])` |
| J8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[deepseek-0528-v1],   Table...` |
| K8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[deepseek-0528-v1], ...` |
| L8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[deepseek-0528-v1],   Table2...` |
| M8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[deepseek-0528-v1],   Table...` |
| N8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3...` |
| O8 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[deepseek-0528-v1],   Table2_3[deepseek-0528-v2],   Table2_3[deepseek-0528-v3] )` |
| B9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3...` |
| C9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss20b (L) v1],   T...` |
| D9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[...` |
| E9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3...` |
| F9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Cl...` |
| G9 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss20b (L) v1],Table1_3[oss20b (L) v2], Table1_3[oss20b (L) v3])` |
| J9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3...` |
| K9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss20b (L) v1],   T...` |
| L9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[...` |
| M9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3...` |
| N9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Cl...` |
| O9 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss20b (L) v1],   Table2_3[oss20b (L) v2],   Table2_3[oss20b (L) v3] )` |
| B10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3...` |
| C10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss20b (M) v1],   T...` |
| D10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[...` |
| E10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3...` |
| F10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Cl...` |
| G10 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss20b (M) v1],Table1_3[oss20b (M) v2], Table1_3[oss20b (M) v3])` |
| J10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3...` |
| K10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss20b (M) v1],   T...` |
| L10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[...` |
| M10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3...` |
| N10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Cl...` |
| O10 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss20b (M) v1],   Table2_3[oss20b (M) v2],   Table2_3[oss20b (M) v3] )` |
| B11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3...` |
| C11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss20b (H) v1],   T...` |
| D11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[...` |
| E11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3...` |
| F11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Cl...` |
| G11 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss20b (H) v1],Table1_3[oss20b (H) v2], Table1_3[oss20b (H) v3])` |
| J11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3...` |
| K11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss20b (H) v1],   T...` |
| L11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[...` |
| M11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3...` |
| N11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Cl...` |
| O11 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss20b (H) v1],   Table2_3[oss20b (H) v2],   Table2_3[oss20b (H) v3] )` |
| B12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss120b (L) v1],   Table1_...` |
| C12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss120b (L) v1],   ...` |
| D12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3...` |
| E12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss120b (L) v1],   Table1_...` |
| F12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[C...` |
| G12 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss120b (L) v1],Table1_3[oss120b (L) v2], Table1_3[oss120b (L) v3])` |
| J12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss120b (L) v1],   Table2_...` |
| K12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss120b (L) v1],   ...` |
| L12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3...` |
| M12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss120b (L) v1],   Table2_...` |
| N12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[C...` |
| O12 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss120b (L) v1],   Table2_3[oss120b (L) v2],   Table2_3[oss120b (L) v3] )` |
| B13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss120b (M) v1],   Table1_...` |
| C13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss120b (M) v1],   ...` |
| D13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3...` |
| E13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss120b (M) v1],   Table1_...` |
| F13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[C...` |
| G13 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss120b (M) v1],Table1_3[oss120b (M) v2], Table1_3[oss120b (M) v3])` |
| J13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss120b (M) v1],   Table2_...` |
| K13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss120b (M) v1],   ...` |
| L13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3...` |
| M13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss120b (M) v1],   Table2_...` |
| N13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[C...` |
| O13 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss120b (M) v1],   Table2_3[oss120b (M) v2],   Table2_3[oss120b (M) v3] )` |
| B14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss120b (H) v1],   Table1_...` |
| C14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss120b (H) v1],   ...` |
| D14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3...` |
| E14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss120b (H) v1],   Table1_...` |
| F14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[C...` |
| G14 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss120b (H) v1],Table1_3[oss120b (H) v2], Table1_3[oss120b (H) v3])` |
| J14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss120b (H) v1],   Table2_...` |
| K14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss120b (H) v1],   ...` |
| L14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3...` |
| M14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss120b (H) v1],   Table2_...` |
| N14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[C...` |
| O14 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss120b (H) v1],   Table2_3[oss120b (H) v2],   Table2_3[oss120b (H) v3] )` |
| B15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[qwen3.5 9B v1],   Table1_3...` |
| C15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[qwen3.5 9B v1],   T...` |
| D15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[qwen3.5 9B v1],   Table1_3[...` |
| E15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[qwen3.5 9B v1],   Table1_3...` |
| F15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[qwen3.5 9B v1],   Table1_3[Cl...` |
| G15 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[qwen3.5 9B v1],Table1_3[qwen3.5 9B v2], Table1_3[qwen3.5 9B v3])` |
| J15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[qwen3.5 9B v1],   Table2_3...` |
| K15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[qwen3.5 9B v1],   T...` |
| L15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[qwen3.5 9B v1],   Table2_3[...` |
| M15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[qwen3.5 9B v1],   Table2_3...` |
| N15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[qwen3.5 9B v1],   Table2_3[Cl...` |
| O15 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table2_3[HumanEvalScore], Table2_3[qwen3.5 9B v1],Table2_3[qwen3.5 9B v2], Table2_3[qwen3.5 9B v3])` |
| B16 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[qwen3.5 27B v1],   Table1_...` |
| C16 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[qwen3.5 27B v1],   ...` |
| D16 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[qwen3.5 27B v1],   Table1_3...` |
| E16 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[qwen3.5 27B v1],   Table1_...` |
| F16 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[qwen3.5 27B v1],   Table1_3[C...` |
| G16 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[qwen3.5 27B v1],Table1_3[qwen3.5 27B v2], Table1_3[qwen3.5 27B v3])` |
| J16 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[qwen3.5 27B v1],   Table2_...` |
| K16 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[qwen3.5 27B v1],   ...` |
| L16 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[qwen3.5 27B v1],   Table2_3...` |
| M16 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[qwen3.5 27B v1],   Table2_...` |
| N16 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[qwen3.5 27B v1],   Table2_3[C...` |
| O16 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table2_3[HumanEvalScore], Table2_3[qwen3.5 27B v1],Table2_3[qwen3.5 27B v2], Table2_3[qwen3.5 27B v3])` |
| B17 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[qwen3.5 35B v1],   Table1_...` |
| C17 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[qwen3.5 35B v1],   ...` |
| D17 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[qwen3.5 35B v1],   Table1_3...` |
| E17 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[qwen3.5 35B v1],   Table1_...` |
| F17 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[qwen3.5 35B v1],   Table1_3[C...` |
| G17 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[qwen3.5 35B v1],Table1_3[qwen3.5 35B v2], Table1_3[qwen3.5 35B v3])` |
| J17 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[qwen3.5 35B v1],   Table2_...` |
| K17 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[qwen3.5 35B v1],   ...` |
| L17 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[qwen3.5 35B v1],   Table2_3...` |
| M17 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[qwen3.5 35B v1],   Table2_...` |
| N17 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[qwen3.5 35B v1],   Table2_3[C...` |
| O17 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table2_3[HumanEvalScore], Table2_3[qwen3.5 35B v1],Table2_3[qwen3.5 35B v2], Table2_3[qwen3.5 35B v3])` |
| B55 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clini...` |
| C55 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_...` |
| D55 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clinic...` |
| E55 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clini...` |
| F55 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clinical...` |
| G55 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[gpt5-0807-m1],Table1_3[gpt5-0807-m2], Table1_3[gpt5-0807-m3])` |
| J55 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[gpt5-0807-m1],   Table2_3[Clini...` |
| K55 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[gpt5-0807-m1],   Table2_...` |
| L55 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[gpt5-0807-m1],   Table2_3[Clinic...` |
| M55 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[gpt5-0807-m1],   Table2_3[Clini...` |
| N55 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[gpt5-0807-m1],   Table2_3[Clinical...` |
| O55 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[gpt5-0807-m1],   Table2_3[gpt5-0807-m2],   Table2_3[gpt5-0807-m3] )` |
| B56 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinica...` |
| C56 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[...` |
| D56 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinical...` |
| E56 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinica...` |
| F56 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinical s...` |
| G56 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[o4-mini-m1],Table1_3[o4-mini-m2], Table1_3[o4-mini-m3])` |
| J56 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[o4-mini-m1],   Table2_3[Clinica...` |
| K56 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[o4-mini-m1],   Table2_3[...` |
| L56 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[o4-mini-m1],   Table2_3[Clinical...` |
| M56 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[o4-mini-m1],   Table2_3[Clinica...` |
| N56 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[o4-mini-m1],   Table2_3[Clinical s...` |
| O56 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[o4-mini-m1],   Table2_3[o4-mini-m2],   Table2_3[o4-mini-m3] )` |
| B57 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_...` |
| C57 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[deepseek-r1-0528-v1],   ...` |
| D57 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3...` |
| E57 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_...` |
| F57 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[C...` |
| G57 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[deepseek-r1-0528-v1],Table1_3[deepseek-r1-0528-v2], Table1_3[deepseek-r1-0528-v3])` |
| J57 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3[C...` |
| K57 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[deepseek-0528-v1],   Tab...` |
| L57 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3[Cl...` |
| M57 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3[C...` |
| N57 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3[Clin...` |
| O57 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[deepseek-0528-v1],   Table2_3[deepseek-0528-v2],   Table2_3[deepseek-0528-v3] )` |
| B58 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clin...` |
| C58 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss20b (L) v1],   Table1...` |
| D58 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clini...` |
| E58 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clin...` |
| F58 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clinica...` |
| G58 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss20b (L) v1],Table1_3[oss20b (L) v2], Table1_3[oss20b (L) v3])` |
| J58 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Clin...` |
| K58 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss20b (L) v1],   Table2...` |
| L58 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Clini...` |
| M58 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Clin...` |
| N58 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Clinica...` |
| O58 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss20b (L) v1],   Table2_3[oss20b (L) v2],   Table2_3[oss20b (L) v3] )` |
| B59 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clin...` |
| C59 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss20b (M) v1],   Table1...` |
| D59 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clini...` |
| E59 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clin...` |
| F59 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clinica...` |
| G59 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss20b (M) v1],Table1_3[oss20b (M) v2], Table1_3[oss20b (M) v3])` |
| J59 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Clin...` |
| K59 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss20b (M) v1],   Table2...` |
| L59 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Clini...` |
| M59 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Clin...` |
| N59 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Clinica...` |
| O59 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss20b (M) v1],   Table2_3[oss20b (M) v2],   Table2_3[oss20b (M) v3] )` |
| B60 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clin...` |
| C60 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss20b (H) v1],   Table1...` |
| D60 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clini...` |
| E60 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clin...` |
| F60 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clinica...` |
| G60 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss20b (H) v1],Table1_3[oss20b (H) v2], Table1_3[oss20b (H) v3])` |
| J60 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Clin...` |
| K60 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss20b (H) v1],   Table2...` |
| L60 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Clini...` |
| M60 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Clin...` |
| N60 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Clinica...` |
| O60 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss20b (H) v1],   Table2_3[oss20b (H) v2],   Table2_3[oss20b (H) v3] )` |
| B61 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Cli...` |
| C61 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss120b (L) v1],   Table...` |
| D61 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Clin...` |
| E61 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Cli...` |
| F61 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Clinic...` |
| G61 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss120b (L) v1],Table1_3[oss120b (L) v2], Table1_3[oss120b (L) v3])` |
| J61 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[Cli...` |
| K61 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss120b (L) v1],   Table...` |
| L61 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[Clin...` |
| M61 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[Cli...` |
| N61 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[Clinic...` |
| O61 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss120b (L) v1],   Table2_3[oss120b (L) v2],   Table2_3[oss120b (L) v3] )` |
| B62 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Cli...` |
| C62 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss120b (M) v1],   Table...` |
| D62 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Clin...` |
| E62 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Cli...` |
| F62 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Clinic...` |
| G62 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss120b (M) v1],Table1_3[oss120b (M) v2], Table1_3[oss120b (M) v3])` |
| J62 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[Cli...` |
| K62 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss120b (M) v1],   Table...` |
| L62 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[Clin...` |
| M62 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[Cli...` |
| N62 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[Clinic...` |
| O62 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss120b (M) v1],   Table2_3[oss120b (M) v2],   Table2_3[oss120b (M) v3] )` |
| B63 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Cli...` |
| C63 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss120b (H) v1],   Table...` |
| D63 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Clin...` |
| E63 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Cli...` |
| F63 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Clinic...` |
| G63 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss120b (H) v1],Table1_3[oss120b (H) v2], Table1_3[oss120b (H) v3])` |
| J63 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[Cli...` |
| K63 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss120b (H) v1],   Table...` |
| L63 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[Clin...` |
| M63 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[Cli...` |
| N63 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[Clinic...` |
| O63 | `=LIKERT_SIGNED_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss120b (H) v1],   Table2_3[oss120b (H) v2],   Table2_3[oss120b (H) v3] )` |
| B71 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clinical special...` |
| C71 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clinical ...` |
| D71 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clinical specialt...` |
| E71 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clinical special...` |
| F71 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clinical specialty]...` |
| G71 | `=LIKERT_CONSENSUS_MAE_CI(   Table1_3[HumanEvalScore],   Table1_3[gpt5-0807-m1],   Table1_3[gpt5-0807-m2],   Table1_3[gpt5-0807-m3] )` |
| J71 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Gynecology""),   FILTER('NMED Diagnosis 3x'!$G$2:$G$720,   'NMED ...` |
| K71 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Internal Medicine""),   FILTER('NMED Diagnosis 3x'!$G$2:$G$720,  ...` |
| L71 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Neurology""),   FILTER('NMED Diagnosis 3x'!$G$2:$G$720,   'NMED D...` |
| M71 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Pediatrics""),   FILTER('NMED Diagnosis 3x'!$G$2:$G$720,   'NMED ...` |
| N71 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Surgery""),   FILTER('NMED Diagnosis 3x'!$G$2:$G$720,   'NMED Dia...` |
| O71 | `=LIKERT_CONSENSUS_MAE_CI(   'NMED Diagnosis 3x'!$F$2:$F$720,   'NMED Diagnosis 3x'!$G$2:$G$720,   'NMED Diagnosis 3x'!$H$2:$H$720,   'NMED Diagnosis 3x'!$I$2:$I$720 )` |
| B72 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinical specialty...` |
| C72 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinical sp...` |
| D72 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinical specialty]...` |
| E72 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinical specialty...` |
| F72 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinical specialty] =...` |
| G72 | `=LIKERT_CONSENSUS_MAE_CI(   Table1_3[HumanEvalScore],   Table1_3[o4-mini-m1],   Table1_3[o4-mini-m2],   Table1_3[o4-mini-m3] )` |
| J72 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Gynecology""),   FILTER('NMED Diagnosis 3x'!$S$2:$S$720,   'NMED ...` |
| K72 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Internal Medicine""),   FILTER('NMED Diagnosis 3x'!$S$2:$S$720,  ...` |
| L72 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Neurology""),   FILTER('NMED Diagnosis 3x'!$S$2:$S$720,   'NMED D...` |
| M72 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Pediatrics""),   FILTER('NMED Diagnosis 3x'!$S$2:$S$720,   'NMED ...` |
| N72 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Surgery""),   FILTER('NMED Diagnosis 3x'!$S$2:$S$720,   'NMED Dia...` |
| O72 | `=LIKERT_CONSENSUS_MAE_CI(   'NMED Diagnosis 3x'!$F$2:$F$720,   'NMED Diagnosis 3x'!$S$2:$S$720,   'NMED Diagnosis 3x'!$T$2:$T$720,   'NMED Diagnosis 3x'!$U$2:$U$720 )` |
| B73 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[Clinical ...` |
| C73 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[Cl...` |
| D73 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[Clinical s...` |
| E73 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[Clinical ...` |
| F73 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[Clinical spe...` |
| G73 | `=LIKERT_CONSENSUS_MAE_CI(   Table1_3[HumanEvalScore],   Table1_3[deepseek-r1-0528-v1],   Table1_3[deepseek-r1-0528-v2],   Table1_3[deepseek-r1-0528-v3] )` |
| J73 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Gynecology""),   FILTER('NMED Diagnosis 3x'!$Y$2:$Y$720,   'NMED ...` |
| K73 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Internal Medicine""),   FILTER('NMED Diagnosis 3x'!$Y$2:$Y$720,  ...` |
| L73 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Neurology""),   FILTER('NMED Diagnosis 3x'!$Y$2:$Y$720,   'NMED D...` |
| M73 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Pediatrics""),   FILTER('NMED Diagnosis 3x'!$Y$2:$Y$720,   'NMED ...` |
| N73 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Surgery""),   FILTER('NMED Diagnosis 3x'!$Y$2:$Y$720,   'NMED Dia...` |
| O73 | `=LIKERT_CONSENSUS_MAE_CI(   'NMED Diagnosis 3x'!$F$2:$F$720,   'NMED Diagnosis 3x'!$Y$2:$Y$720,   'NMED Diagnosis 3x'!$Z$2:$Z$720,   'NMED Diagnosis 3x'!$AA$2:$AA$720 )` |
| B74 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clinical specia...` |
| C74 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clinical...` |
| D74 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clinical special...` |
| E74 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clinical specia...` |
| F74 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clinical specialty...` |
| G74 | `=LIKERT_CONSENSUS_MAE_CI(   Table1_3[HumanEvalScore],   Table1_3[oss20b (L) v1],   Table1_3[oss20b (L) v2],   Table1_3[oss20b (L) v3] )` |
| J74 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Gynecology""),   FILTER('NMED Diagnosis 3x'!$AB$2:$AB$720,   'NME...` |
| K74 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Internal Medicine""),   FILTER('NMED Diagnosis 3x'!$AB$2:$AB$720,...` |
| L74 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Neurology""),   FILTER('NMED Diagnosis 3x'!$AB$2:$AB$720,   'NMED...` |
| M74 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Pediatrics""),   FILTER('NMED Diagnosis 3x'!$AB$2:$AB$720,   'NME...` |
| N74 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Surgery""),   FILTER('NMED Diagnosis 3x'!$AB$2:$AB$720,   'NMED D...` |
| O74 | `=LIKERT_CONSENSUS_MAE_CI(   'NMED Diagnosis 3x'!$F$2:$F$720,   'NMED Diagnosis 3x'!$AB$2:$AB$720,   'NMED Diagnosis 3x'!$AC$2:$AC$720,   'NMED Diagnosis 3x'!$AD$2:$AD$720 )` |
| B75 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clinical specia...` |
| C75 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clinical...` |
| D75 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clinical special...` |
| E75 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clinical specia...` |
| F75 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clinical specialty...` |
| G75 | `=LIKERT_CONSENSUS_MAE_CI(   Table1_3[HumanEvalScore],   Table1_3[oss20b (M) v1],   Table1_3[oss20b (M) v2],   Table1_3[oss20b (M) v3] )` |
| J75 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Gynecology""),   FILTER('NMED Diagnosis 3x'!$AE$2:$AE$720,   'NME...` |
| K75 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Internal Medicine""),   FILTER('NMED Diagnosis 3x'!$AE$2:$AE$720,...` |
| L75 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Neurology""),   FILTER('NMED Diagnosis 3x'!$AE$2:$AE$720,   'NMED...` |
| M75 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Pediatrics""),   FILTER('NMED Diagnosis 3x'!$AE$2:$AE$720,   'NME...` |
| N75 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Surgery""),   FILTER('NMED Diagnosis 3x'!$AE$2:$AE$720,   'NMED D...` |
| O75 | `=LIKERT_CONSENSUS_MAE_CI(   'NMED Diagnosis 3x'!$F$2:$F$720,   'NMED Diagnosis 3x'!$AE$2:$AE$720,   'NMED Diagnosis 3x'!$AF$2:$AF$720,   'NMED Diagnosis 3x'!$AG$2:$AG$720 )` |
| B76 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clinical specia...` |
| C76 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clinical...` |
| D76 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clinical special...` |
| E76 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clinical specia...` |
| F76 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clinical specialty...` |
| G76 | `=LIKERT_CONSENSUS_MAE_CI(   Table1_3[HumanEvalScore],   Table1_3[oss20b (H) v1],   Table1_3[oss20b (H) v2],   Table1_3[oss20b (H) v3] )` |
| J76 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Gynecology""),   FILTER('NMED Diagnosis 3x'!$AH$2:$AH$720,   'NME...` |
| K76 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Internal Medicine""),   FILTER('NMED Diagnosis 3x'!$AH$2:$AH$720,...` |
| L76 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Neurology""),   FILTER('NMED Diagnosis 3x'!$AH$2:$AH$720,   'NMED...` |
| M76 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Pediatrics""),   FILTER('NMED Diagnosis 3x'!$AH$2:$AH$720,   'NME...` |
| N76 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Surgery""),   FILTER('NMED Diagnosis 3x'!$AH$2:$AH$720,   'NMED D...` |
| O76 | `=LIKERT_CONSENSUS_MAE_CI(   'NMED Diagnosis 3x'!$F$2:$F$720,   'NMED Diagnosis 3x'!$AH$2:$AH$720,   'NMED Diagnosis 3x'!$AI$2:$AI$720,   'NMED Diagnosis 3x'!$AJ$2:$AJ$720 )` |
| B77 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Clinical speci...` |
| C77 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Clinica...` |
| D77 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Clinical specia...` |
| E77 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Clinical speci...` |
| F77 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Clinical specialt...` |
| G77 | `=LIKERT_CONSENSUS_MAE_CI(   Table1_3[HumanEvalScore],   Table1_3[oss120b (L) v1],   Table1_3[oss120b (L) v2],   Table1_3[oss120b (L) v3] )` |
| J77 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Gynecology""),   FILTER('NMED Diagnosis 3x'!$AK$2:$AK$720,   'NME...` |
| K77 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Internal Medicine""),   FILTER('NMED Diagnosis 3x'!$AK$2:$AK$720,...` |
| L77 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Neurology""),   FILTER('NMED Diagnosis 3x'!$AK$2:$AK$720,   'NMED...` |
| M77 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Pediatrics""),   FILTER('NMED Diagnosis 3x'!$AK$2:$AK$720,   'NME...` |
| N77 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Surgery""),   FILTER('NMED Diagnosis 3x'!$AK$2:$AK$720,   'NMED D...` |
| O77 | `=LIKERT_CONSENSUS_MAE_CI(   'NMED Diagnosis 3x'!$F$2:$F$720,   'NMED Diagnosis 3x'!$AK$2:$AK$720,   'NMED Diagnosis 3x'!$AL$2:$AL$720,   'NMED Diagnosis 3x'!$AM$2:$AM$720 )` |
| B78 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Clinical speci...` |
| C78 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Clinica...` |
| D78 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Clinical specia...` |
| E78 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Clinical speci...` |
| F78 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Clinical specialt...` |
| G78 | `=LIKERT_CONSENSUS_MAE_CI(   Table1_3[HumanEvalScore],   Table1_3[oss120b (M) v1],   Table1_3[oss120b (M) v2],   Table1_3[oss120b (M) v3] )` |
| J78 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Gynecology""),   FILTER('NMED Diagnosis 3x'!$AN$2:$AN$720,   'NME...` |
| K78 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Internal Medicine""),   FILTER('NMED Diagnosis 3x'!$AN$2:$AN$720,...` |
| L78 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Neurology""),   FILTER('NMED Diagnosis 3x'!$AN$2:$AN$720,   'NMED...` |
| M78 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Pediatrics""),   FILTER('NMED Diagnosis 3x'!$AN$2:$AN$720,   'NME...` |
| N78 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Surgery""),   FILTER('NMED Diagnosis 3x'!$AN$2:$AN$720,   'NMED D...` |
| O78 | `=LIKERT_CONSENSUS_MAE_CI(   'NMED Diagnosis 3x'!$F$2:$F$720,   'NMED Diagnosis 3x'!$AN$2:$AN$720,   'NMED Diagnosis 3x'!$AO$2:$AO$720,   'NMED Diagnosis 3x'!$AP$2:$AP$720 )` |
| B79 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Clinical speci...` |
| C79 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Clinica...` |
| D79 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Clinical specia...` |
| E79 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Clinical speci...` |
| F79 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Clinical specialt...` |
| G79 | `=LIKERT_CONSENSUS_MAE_CI(   Table1_3[HumanEvalScore],   Table1_3[oss120b (H) v1],   Table1_3[oss120b (H) v2],   Table1_3[oss120b (H) v3] )` |
| J79 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Gynecology""),   FILTER('NMED Diagnosis 3x'!$AQ$2:$AQ$720,   'NME...` |
| K79 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Internal Medicine""),   FILTER('NMED Diagnosis 3x'!$AQ$2:$AQ$720,...` |
| L79 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Neurology""),   FILTER('NMED Diagnosis 3x'!$AQ$2:$AQ$720,   'NMED...` |
| M79 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Pediatrics""),   FILTER('NMED Diagnosis 3x'!$AQ$2:$AQ$720,   'NME...` |
| N79 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER('NMED Diagnosis 3x'!$F$2:$F$720, 'NMED Diagnosis 3x'!$A$2:$A$720 = ""Surgery""),   FILTER('NMED Diagnosis 3x'!$AQ$2:$AQ$720,   'NMED D...` |
| O79 | `=LIKERT_CONSENSUS_MAE_CI(   'NMED Diagnosis 3x'!$F$2:$F$720,   'NMED Diagnosis 3x'!$AQ$2:$AQ$720,   'NMED Diagnosis 3x'!$AR$2:$AR$720,   'NMED Diagnosis 3x'!$AS$2:$AS$720 )` |

### nmed_frequency (104 formulas)

| Cell | Formula |
|------|--------|
| C3 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[gpt-5.1-1113-m1],Table1_3[gpt-5.1-1113-m2], Table1_3[gpt-5.1-1113-m3])` |
| D3 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[gpt-5.1-1113-m1],   Table1_3[Frequency] =...` |
| E3 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[gpt-5.1-1113-m1],   Table1_3[Fre...` |
| F3 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[gpt-5.1-1113-m1],   Table1_3[Frequenc...` |
| C4 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[gemini-3.1-pro-m1],Table1_3[gemini-3.1-pro-m2], Table1_3[gemini-3.1-pro-m3])` |
| D4 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[gemini-3.1-pro-m1],   Table1_3[Frequency]...` |
| E4 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[gemini-3.1-pro-m1],   Table1_3[F...` |
| F4 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[gemini-3.1-pro-m1],   Table1_3[Freque...` |
| C5 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[gpt-5-mini-0807-m1],Table1_3[gpt-5-mini-0807-m2], Table1_3[gpt-5-mini-0807-m3])` |
| D5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[gpt-5-mini-0807-m1],   Table1_3[Frequency...` |
| E5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[gpt-5-mini-0807-m1],   Table1_3[...` |
| F5 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[gpt-5-mini-0807-m1],   Table1_3[Frequ...` |
| C6 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[deepseek-r1-0528-v1],Table1_3[deepseek-r1-0528-v2], Table1_3[deepseek-r1-0528-v3])` |
| D6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[Frequenc...` |
| E6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3...` |
| F6 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[Freq...` |
| C7 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss20b (L) v1],Table1_3[oss20b (L) v2], Table1_3[oss20b (L) v3])` |
| D7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Frequency] = "...` |
| E7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Frequ...` |
| F7 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Frequency]...` |
| C8 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss20b (M) v1],Table1_3[oss20b (M) v2], Table1_3[oss20b (M) v3])` |
| D8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Frequency] = "...` |
| E8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Frequ...` |
| F8 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Frequency]...` |
| C9 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss20b (H) v1],Table1_3[oss20b (H) v2], Table1_3[oss20b (H) v3])` |
| D9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Frequency] = "...` |
| E9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Frequ...` |
| F9 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Frequency]...` |
| C10 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss120b (L) v1],Table1_3[oss120b (L) v2], Table1_3[oss120b (L) v3])` |
| D10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Frequency] = ...` |
| E10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Freq...` |
| F10 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Frequency...` |
| C11 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss120b (M) v1],Table1_3[oss120b (M) v2], Table1_3[oss120b (M) v3])` |
| D11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Frequency] = ...` |
| E11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Freq...` |
| F11 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Frequency...` |
| C12 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[oss120b (H) v1],Table1_3[oss120b (H) v2], Table1_3[oss120b (H) v3])` |
| D12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Frequency] = ...` |
| E12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Freq...` |
| F12 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Frequency...` |
| C13 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[qwen3.5 9B v1],Table1_3[qwen3.5 9B v2], Table1_3[qwen3.5 9B v3])` |
| D13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[qwen3.5 9B v1],   Table1_3[Frequency] = "...` |
| E13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[qwen3.5 9B v1],   Table1_3[Frequ...` |
| F13 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[qwen3.5 9B v1],   Table1_3[Frequency]...` |
| C14 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[qwen3.5 27B v1],Table1_3[qwen3.5 27B v2], Table1_3[qwen3.5 27B v3])` |
| D14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[qwen3.5 27B v1],   Table1_3[Frequency] = ...` |
| E14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[qwen3.5 27B v1],   Table1_3[Freq...` |
| F14 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[qwen3.5 27B v1],   Table1_3[Frequency...` |
| C15 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table1_3[HumanEvalScore], Table1_3[qwen3.5 35B v1],Table1_3[qwen3.5 35B v2], Table1_3[qwen3.5 35B v3])` |
| D15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Rare""),   FILTER(Table1_3[qwen3.5 35B v1],   Table1_3[Frequency] = ...` |
| E15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Less Frequent""),   FILTER(Table1_3[qwen3.5 35B v1],   Table1_3[Freq...` |
| F15 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table1_3[HumanEvalScore], Table1_3[Frequency] = ""Frequent""),   FILTER(Table1_3[qwen3.5 35B v1],   Table1_3[Frequency...` |
| C20 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[gpt-5.1-1113-m1],   Table2_3[gpt-5.1-1113-m2],   Table2_3[gpt-5.1-1113-m3] )` |
| D20 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[gpt-5.1-1113-m1],   Table2_3[Frequency] =...` |
| E20 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[gpt-5.1-1113-m1],   Table2_3[Fre...` |
| F20 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[gpt-5.1-1113-m1],   Table2_3[Frequenc...` |
| C21 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[gemini-3.1-pro-m1],   Table2_3[gemini-3.1-pro-m2],   Table2_3[gemini-3.1-pro-m3] )` |
| D21 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[gemini-3.1-pro-m1],   Table2_3[Frequency]...` |
| E21 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[gemini-3.1-pro-m1],   Table2_3[F...` |
| F21 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[gemini-3.1-pro-m1],   Table2_3[Freque...` |
| C22 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[gpt-5-mini-0807-m1],   Table2_3[gpt-5-mini-0807-m2],   Table2_3[gpt-5-mini-0807-m3] )` |
| D22 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[gpt-5-mini-0807-m1],   Table2_3[Frequency...` |
| E22 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[gpt-5-mini-0807-m1],   Table2_3[...` |
| F22 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[gpt-5-mini-0807-m1],   Table2_3[Frequ...` |
| C23 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[deepseek-0528-v1],   Table2_3[deepseek-0528-v2],   Table2_3[deepseek-0528-v3] )` |
| D23 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3[Frequency] ...` |
| E23 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3[Fr...` |
| F23 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3[Frequen...` |
| C24 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss20b (L) v1],   Table2_3[oss20b (L) v2],   Table2_3[oss20b (L) v3] )` |
| D24 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Frequency] = "...` |
| E24 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Frequ...` |
| F24 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Frequency]...` |
| C25 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss20b (M) v1],   Table2_3[oss20b (M) v2],   Table2_3[oss20b (M) v3] )` |
| D25 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Frequency] = "...` |
| E25 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Frequ...` |
| F25 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Frequency]...` |
| C26 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss20b (H) v1],   Table2_3[oss20b (H) v2],   Table2_3[oss20b (H) v3] )` |
| D26 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Frequency] = "...` |
| E26 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Frequ...` |
| F26 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Frequency]...` |
| C27 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss120b (L) v1],   Table2_3[oss120b (L) v2],   Table2_3[oss120b (L) v3] )` |
| D27 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[Frequency] = ...` |
| E27 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[Freq...` |
| F27 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[Frequency...` |
| C28 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss120b (M) v1],   Table2_3[oss120b (M) v2],   Table2_3[oss120b (M) v3] )` |
| D28 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[Frequency] = ...` |
| E28 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[Freq...` |
| F28 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[Frequency...` |
| C29 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   Table2_3[HumanEvalScore],   Table2_3[oss120b (H) v1],   Table2_3[oss120b (H) v2],   Table2_3[oss120b (H) v3] )` |
| D29 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[Frequency] = ...` |
| E29 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[Freq...` |
| F29 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[Frequency...` |
| C30 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table2_3[HumanEvalScore], Table2_3[qwen3.5 9B v1],Table2_3[qwen3.5 9B v2], Table2_3[qwen3.5 9B v3])` |
| D30 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[qwen3.5 9B v1],   Table2_3[Frequency] = "...` |
| E30 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[qwen3.5 9B v1],   Table2_3[Frequ...` |
| F30 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[qwen3.5 9B v1],   Table2_3[Frequency]...` |
| C31 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table2_3[HumanEvalScore], Table2_3[qwen3.5 27B v1],Table2_3[qwen3.5 27B v2], Table2_3[qwen3.5 27B v3])` |
| D31 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[qwen3.5 27B v1],   Table2_3[Frequency] = ...` |
| E31 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[qwen3.5 27B v1],   Table2_3[Freq...` |
| F31 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[qwen3.5 27B v1],   Table2_3[Frequency...` |
| C32 | `=LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(Table2_3[HumanEvalScore], Table2_3[qwen3.5 35B v1],Table2_3[qwen3.5 35B v2], Table2_3[qwen3.5 35B v3])` |
| D32 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Rare""),   FILTER(Table2_3[qwen3.5 35B v1],   Table2_3[Frequency] = ...` |
| E32 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Less Frequent""),   FILTER(Table2_3[qwen3.5 35B v1],   Table2_3[Freq...` |
| F32 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_SIGNED_MEAN_CONSENSUS_MEDIAN_IQR(   FILTER(Table2_3[HumanEvalScore], Table2_3[Frequency] = ""Frequent""),   FILTER(Table2_3[qwen3.5 35B v1],   Table2_3[Frequency...` |

### NMED Diagnosis 3x (144 formulas)

| Cell | Formula |
|------|--------|
| G722 | `=WILCOXON_CONSENSUS_TEST(F2:F721, G2:I721, S2:U721)` |
| Z722 | `=MAE(   INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),   Z2:Z721 )` |
| AA722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AA2:AA721)` |
| AB722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AB2:AB721)` |
| AC722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AC2:AC721)` |
| AD722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AD2:AD721)` |
| AE722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AE2:AE721)` |
| AF722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AF2:AF721)` |
| AG722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AG2:AG721)` |
| AH722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AH2:AH721)` |
| AI722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AI2:AI721)` |
| AJ722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AJ2:AJ721)` |
| AK722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AK2:AK721)` |
| AL722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AL2:AL721)` |
| AM722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AM2:AM721)` |
| AN722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AN2:AN721)` |
| AO722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AO2:AO721)` |
| AP722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AP2:AP721)` |
| AQ722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AQ2:AQ721)` |
| AR722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AR2:AR721)` |
| AS722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AS2:AS721)` |
| AT722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AT2:AT721)` |
| AU722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AU2:AU721)` |
| AV722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AV2:AV721)` |
| AW722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AW2:AW721)` |
| AX722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AX2:AX721)` |
| AY722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AY2:AY721)` |
| AZ722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  AZ2:AZ721)` |
| BA722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  BA2:BA721)` |
| BB722 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  BB2:BB721)` |
| Z723 | `=AVERAGE(Y722:AB722)` |
| F728 | `=LIKERT_CONSENSUS_MAE_CI(Table2_3[HumanEvalScore],Table2_3[gpt5-0807-m1], Table2_3[gpt5-0807-m2],Table2_3[gpt5-0807-m3])` |
| S728 | `=LIKERT_CONSENSUS_MAE_CI(Table2_3[HumanEvalScore],Table2_3[o4-mini-m1], Table2_3[o4-mini-m2],Table2_3[o4-mini-m3])` |
| Y728 | `=LIKERT_CONSENSUS_MAE_CI(Table2_3[HumanEvalScore],Table2_3[deepseek-0528-v1], Table2_3[deepseek-0528-v2],Table2_3[deepseek-0528-v3])` |
| AB728 | `=LIKERT_CONSENSUS_MAE_CI(Table2_3[HumanEvalScore],Table2_3[oss20b (L) v1], Table2_3[oss20b (L) v2],Table2_3[oss20b (L) v3])` |
| AE728 | `=LIKERT_CONSENSUS_MAE_CI(Table2_3[HumanEvalScore],Table2_3[oss20b (M) v1], Table2_3[oss20b (M) v2],Table2_3[oss20b (M) v3])` |
| AH728 | `=LIKERT_CONSENSUS_MAE_CI(Table2_3[HumanEvalScore],Table2_3[oss20b (H) v1], Table2_3[oss20b (H) v2],Table2_3[oss20b (H) v3])` |
| AK728 | `=LIKERT_CONSENSUS_MAE_CI(Table2_3[HumanEvalScore],Table2_3[oss120b (L) v1], Table2_3[oss120b (L) v2],Table2_3[oss120b (L) v3])` |
| AN728 | `=LIKERT_CONSENSUS_MAE_CI(Table2_3[HumanEvalScore],Table2_3[oss120b (M) v1], Table2_3[oss120b (M) v2],Table2_3[oss120b (M) v3])` |
| AQ728 | `=LIKERT_CONSENSUS_MAE_CI(Table2_3[HumanEvalScore],Table2_3[oss120b (H) v1], Table2_3[oss120b (H) v2],Table2_3[oss120b (H) v3])` |
| G731 | `=MAE(   INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),   G2:G721 )` |
| H731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  H2:H721)` |
| I731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  I2:I721)` |
| J731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  J2:J721)` |
| K731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  K2:K721)` |
| L731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  L2:L721)` |
| M731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  M2:M721)` |
| N731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  N2:N721)` |
| O731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  O2:O721)` |
| P731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  P2:P721)` |
| Q731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  Q2:Q721)` |
| R731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  R2:R721)` |
| S731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  S2:S721)` |
| T731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  T2:T721)` |
| U731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  U2:U721)` |
| V731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  V2:V721)` |
| W731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  W2:W721)` |
| X731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  X2:X721)` |
| Y731 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  Y2:Y721)` |
| G735 | `=MAE_IQR_TEXT($F$2:$F$721, G2:G721) ` |
| Y735 | `=MAE_IQR_TEXT($F$2:$F$721, Y2:Y721) ` |
| AB735 | `=MAE_IQR_TEXT($F$2:$F$721, AB2:AB721) ` |
| AE735 | `=MAE_IQR_TEXT($F$2:$F$721, AE2:AE721) ` |
| AH735 | `=MAE_IQR_TEXT($F$2:$F$721, AH2:AH721) ` |
| AK735 | `=MAE_IQR_TEXT($F$2:$F$721, AK2:AK721) ` |
| AN735 | `=MAE_IQR_TEXT($F$2:$F$721, AN2:AN721) ` |
| AQ735 | `=MAE_IQR_TEXT($F$2:$F$721, AQ2:AQ721) ` |
| G736 | `=MEDAE_IQR_TEXT($F$2:$F$721, G2:G721)` |
| Y736 | `=MEDAE_IQR_TEXT($F$2:$F$721, Y2:Y721)` |
| AB736 | `=MEDAE_IQR_TEXT($F$2:$F$721, AB2:AB721)` |
| AE736 | `=MEDAE_IQR_TEXT($F$2:$F$721, AE2:AE721)` |
| AH736 | `=MEDAE_IQR_TEXT($F$2:$F$721, AH2:AH721)` |
| AK736 | `=MEDAE_IQR_TEXT($F$2:$F$721, AK2:AK721)` |
| AN736 | `=MEDAE_IQR_TEXT($F$2:$F$721, AN2:AN721)` |
| AQ736 | `=MEDAE_IQR_TEXT($F$2:$F$721, AQ2:AQ721)` |
| G737 | `=MEAN_SCORE_CI_TEXT(G2:G721)` |
| Y737 | `=MEAN_SCORE_CI_TEXT(Y2:Y721)` |
| AB737 | `=MEAN_SCORE_CI_TEXT(AB2:AB721)` |
| AE737 | `=MEAN_SCORE_CI_TEXT(AE2:AE721)` |
| AH737 | `=MEAN_SCORE_CI_TEXT(AH2:AH721)` |
| AK737 | `=MEAN_SCORE_CI_TEXT(AK2:AK721)` |
| AN737 | `=MEAN_SCORE_CI_TEXT(AN2:AN721)` |
| AQ737 | `=MEAN_SCORE_CI_TEXT(AQ2:AQ721)` |
| G738 | `=MAE_CI_TEXT($F$2:$F$721, G2:G721) ` |
| Y738 | `=MAE_CI_TEXT($F$2:$F$721, Y2:Y721) ` |
| AB738 | `=MAE_CI_TEXT($F$2:$F$721, AB2:AB721) ` |
| AE738 | `=MAE_CI_TEXT($F$2:$F$721, AE2:AE721) ` |
| AH738 | `=MAE_CI_TEXT($F$2:$F$721, AH2:AH721) ` |
| AK738 | `=MAE_CI_TEXT($F$2:$F$721, AK2:AK721) ` |
| AN738 | `=MAE_CI_TEXT($F$2:$F$721, AN2:AN721) ` |
| AQ738 | `=MAE_CI_TEXT($F$2:$F$721, AQ2:AQ721) ` |
| G740 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, G2:G721)` |
| Y740 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, Y2:Y721)` |
| AB740 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, AB2:AB721)` |
| AE740 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, AE2:AE721)` |
| AH740 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, AH2:AH721)` |
| AK740 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, AK2:AK721)` |
| AN740 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, AN2:AN721)` |
| AQ740 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, AQ2:AQ721)` |
| AJ747 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[gpt5-0807-m1],   Table2_3[Clinical special...` |
| AK747 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[gpt5-0807-m1],   Table2_3[Clinical ...` |
| AL747 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[gpt5-0807-m1],   Table2_3[Clinical specialt...` |
| AM747 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[gpt5-0807-m1],   Table2_3[Clinical special...` |
| AN747 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[gpt5-0807-m1],   Table2_3[Clinical specialty]...` |
| AJ748 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[o4-mini-m1],   Table2_3[Clinical specialty...` |
| AK748 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[o4-mini-m1],   Table2_3[Clinical sp...` |
| AL748 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[o4-mini-m1],   Table2_3[Clinical specialty]...` |
| AM748 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[o4-mini-m1],   Table2_3[Clinical specialty...` |
| AN748 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[o4-mini-m1],   Table2_3[Clinical specialty] =...` |
| AJ749 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3[Clinical spe...` |
| AK749 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3[Clini...` |
| AL749 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3[Clinical spec...` |
| AM749 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3[Clinical spe...` |
| AN749 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[deepseek-0528-v1],   Table2_3[Clinical specia...` |
| AJ750 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Clinical specia...` |
| AK750 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Clinical...` |
| AL750 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Clinical special...` |
| AM750 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Clinical specia...` |
| AN750 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss20b (L) v1],   Table2_3[Clinical specialty...` |
| AJ751 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Clinical specia...` |
| AK751 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Clinical...` |
| AL751 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Clinical special...` |
| AM751 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Clinical specia...` |
| AN751 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss20b (M) v1],   Table2_3[Clinical specialty...` |
| AJ752 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Clinical specia...` |
| AK752 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Clinical...` |
| AL752 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Clinical special...` |
| AM752 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Clinical specia...` |
| AN752 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss20b (H) v1],   Table2_3[Clinical specialty...` |
| AJ753 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[Clinical speci...` |
| AK753 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[Clinica...` |
| AL753 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[Clinical specia...` |
| AM753 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[Clinical speci...` |
| AN753 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss120b (L) v1],   Table2_3[Clinical specialt...` |
| AJ754 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[Clinical speci...` |
| AK754 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[Clinica...` |
| AL754 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[Clinical specia...` |
| AM754 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[Clinical speci...` |
| AN754 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss120b (M) v1],   Table2_3[Clinical specialt...` |
| AJ755 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Gynecology""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[Clinical speci...` |
| AK755 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[Clinica...` |
| AL755 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Neurology""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[Clinical specia...` |
| AM755 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[Clinical speci...` |
| AN755 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table2_3[HumanEvalScore], Table2_3[Clinical specialty] = ""Surgery""),   FILTER(Table2_3[oss120b (H) v1],   Table2_3[Clinical specialt...` |

### NMED Treatment 3x (124 formulas)

| Cell | Formula |
|------|--------|
| G598 | `=MAE(   INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),   G2:G596 )` |
| H598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  H2:H596)` |
| I598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  I2:I596)` |
| J598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  J2:J596)` |
| K598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  K2:K596)` |
| L598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  L2:L596)` |
| M598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  M2:M596)` |
| N598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  N2:N596)` |
| O598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  O2:O596)` |
| P598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  P2:P596)` |
| Q598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  Q2:Q596)` |
| R598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  R2:R596)` |
| S598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  S2:S596)` |
| T598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  T2:T596)` |
| U598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  U2:U596)` |
| V598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  V2:V596)` |
| W598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  W2:W596)` |
| X598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  X2:X596)` |
| Y598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  Y2:Y596)` |
| Z598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  Z2:Z596)` |
| AA598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  AA2:AA596)` |
| AQ598 | `=MAE(   INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),   AQ2:AQ596 )` |
| AR598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  AR2:AR596)` |
| AS598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  AS2:AS596)` |
| AT598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  AT2:AT596)` |
| AU598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  AU2:AU596)` |
| AV598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  AV2:AV596)` |
| AW598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  AW2:AW596)` |
| AX598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  AX2:AX596)` |
| AY598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  AY2:AY596)` |
| AZ598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  AZ2:AZ596)` |
| BA598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  BA2:BA596)` |
| BB598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  BB2:BB596)` |
| Z599 | `=AVERAGE(Y598:AA598)` |
| F601 | `=LIKERT_CONSENSUS_MAE_CI(Table1_3[HumanEvalScore],Table1_3[gpt5-0807-m1], Table1_3[gpt5-0807-m2],Table1_3[gpt5-0807-m3])` |
| S601 | `=LIKERT_CONSENSUS_MAE_CI(Table1_3[HumanEvalScore],Table1_3[o4-mini-m1], Table1_3[o4-mini-m2],Table1_3[o4-mini-m3])` |
| Y601 | `=LIKERT_CONSENSUS_MAE_CI(Table1_3[HumanEvalScore],Table1_3[deepseek-r1-0528-v1], Table1_3[deepseek-r1-0528-v2],Table1_3[deepseek-r1-0528-v3])` |
| AB601 | `=LIKERT_CONSENSUS_MAE_CI(Table1_3[HumanEvalScore],Table1_3[oss20b (L) v1], Table1_3[oss20b (L) v2],Table1_3[oss20b (L) v3])` |
| AE601 | `=LIKERT_CONSENSUS_MAE_CI(Table1_3[HumanEvalScore],Table1_3[oss20b (M) v1], Table1_3[oss20b (M) v2],Table1_3[oss20b (M) v3])` |
| AH601 | `=LIKERT_CONSENSUS_MAE_CI(Table1_3[HumanEvalScore],Table1_3[oss20b (H) v1], Table1_3[oss20b (H) v2],Table1_3[oss20b (H) v3])` |
| AK601 | `=LIKERT_CONSENSUS_MAE_CI(Table1_3[HumanEvalScore],Table1_3[oss120b (L) v1], Table1_3[oss120b (L) v2],Table1_3[oss120b (L) v3])` |
| AN601 | `=LIKERT_CONSENSUS_MAE_CI(Table1_3[HumanEvalScore],Table1_3[oss120b (M) v1], Table1_3[oss120b (M) v2],Table1_3[oss120b (M) v3])` |
| AQ601 | `=LIKERT_CONSENSUS_MAE_CI(Table1_3[HumanEvalScore],Table1_3[oss120b (H) v1], Table1_3[oss120b (H) v2],Table1_3[oss120b (H) v3])` |
| G603 | `=MAE_IQR_TEXT($F$2:$F$596, G2:G596) ` |
| S603 | `=MAE_IQR_TEXT($F$2:$F$596, S2:S596) ` |
| Y603 | `=MAE_IQR_TEXT($F$2:$F$596, Y2:Y596) ` |
| AH603 | `=MAE_IQR_TEXT($F$2:$F$596, AH2:AH596) ` |
| AK603 | `=MAE_IQR_TEXT($F$2:$F$596, AK2:AK596) ` |
| AN603 | `=MAE_IQR_TEXT($F$2:$F$596, AN2:AN596) ` |
| AQ603 | `=MAE_IQR_TEXT($F$2:$F$596, AQ2:AQ596) ` |
| G604 | `=MEDAE_IQR_TEXT($F$2:$F$596, G2:G596)` |
| S604 | `=MEDAE_IQR_TEXT($F$2:$F$596, S2:S596)` |
| Y604 | `=MEDAE_IQR_TEXT($F$2:$F$596, Y2:Y596)` |
| AH604 | `=MEDAE_IQR_TEXT($F$2:$F$596, AH2:AH596)` |
| AK604 | `=MEDAE_IQR_TEXT($F$2:$F$596, AK2:AK596)` |
| AN604 | `=MEDAE_IQR_TEXT($F$2:$F$596, AN2:AN596)` |
| AQ604 | `=MEDAE_IQR_TEXT($F$2:$F$596, AQ2:AQ596)` |
| G605 | `=MEAN_SCORE_CI_TEXT(G2:G596)` |
| S605 | `=MEAN_SCORE_CI_TEXT(S2:S596)` |
| Y605 | `=MEAN_SCORE_CI_TEXT(Y2:Y596)` |
| AH605 | `=MEAN_SCORE_CI_TEXT(AH2:AH596)` |
| AK605 | `=MEAN_SCORE_CI_TEXT(AK2:AK596)` |
| AN605 | `=MEAN_SCORE_CI_TEXT(AN2:AN596)` |
| AQ605 | `=MEAN_SCORE_CI_TEXT(AQ2:AQ596)` |
| G606 | `=MAE_CI_TEXT($F$2:$F$596, G2:G596) ` |
| S606 | `=MAE_CI_TEXT($F$2:$F$596, S2:S596) ` |
| Y606 | `=MAE_CI_TEXT($F$2:$F$596, Y2:Y596) ` |
| AH606 | `=MAE_CI_TEXT($F$2:$F$596, AH2:AH596) ` |
| AK606 | `=MAE_CI_TEXT($F$2:$F$596, AK2:AK596) ` |
| AN606 | `=MAE_CI_TEXT($F$2:$F$596, AN2:AN596) ` |
| AQ606 | `=MAE_CI_TEXT($F$2:$F$596, AQ2:AQ596) ` |
| G608 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, G2:G596)` |
| S608 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, S2:S596)` |
| Y608 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, Y2:Y596)` |
| AH608 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, AH2:AH596)` |
| AK608 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, AK2:AK596)` |
| AN608 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, AN2:AN596)` |
| AQ608 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, AQ2:AQ596)` |
| G612 | `=WILCOXON_ERROR_TEST(F2:F596, G2:G596, S2:S596)` |
| I621 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clinical special...` |
| S621 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clinical ...` |
| T621 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clinical specialt...` |
| U621 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clinical special...` |
| Y621 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[gpt5-0807-m1],   Table1_3[Clinical specialty]...` |
| I622 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinical specialty...` |
| S622 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinical sp...` |
| T622 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinical specialty]...` |
| U622 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinical specialty...` |
| Y622 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[o4-mini-m1],   Table1_3[Clinical specialty] =...` |
| I623 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[Clinical ...` |
| S623 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[Cl...` |
| T623 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[Clinical s...` |
| U623 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[Clinical ...` |
| Y623 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[deepseek-r1-0528-v1],   Table1_3[Clinical spe...` |
| I624 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clinical specia...` |
| S624 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clinical...` |
| T624 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clinical special...` |
| U624 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clinical specia...` |
| Y624 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss20b (L) v1],   Table1_3[Clinical specialty...` |
| I625 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clinical specia...` |
| S625 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clinical...` |
| T625 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clinical special...` |
| U625 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clinical specia...` |
| Y625 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss20b (M) v1],   Table1_3[Clinical specialty...` |
| I626 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clinical specia...` |
| S626 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clinical...` |
| T626 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clinical special...` |
| U626 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clinical specia...` |
| Y626 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss20b (H) v1],   Table1_3[Clinical specialty...` |
| I627 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Clinical speci...` |
| S627 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Clinica...` |
| T627 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Clinical specia...` |
| U627 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Clinical speci...` |
| Y627 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss120b (L) v1],   Table1_3[Clinical specialt...` |
| I628 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Clinical speci...` |
| S628 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Clinica...` |
| T628 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Clinical specia...` |
| U628 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Clinical speci...` |
| Y628 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss120b (M) v1],   Table1_3[Clinical specialt...` |
| I629 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Gynecology""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Clinical speci...` |
| S629 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Internal Medicine""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Clinica...` |
| T629 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Neurology""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Clinical specia...` |
| U629 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Pediatrics""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Clinical speci...` |
| Y629 | `=IFERROR(__xludf.DUMMYFUNCTION("LIKERT_CONSENSUS_MAE_CI(   FILTER(Table1_3[HumanEvalScore], Table1_3[Clinical specialty] = ""Surgery""),   FILTER(Table1_3[oss120b (H) v1],   Table1_3[Clinical specialt...` |

### Eurorad (110 formulas)

| Cell | Formula |
|------|--------|
| C210 | `=COUNTIF($B$2:$B$208, B210)` |
| K210 | `=MODEL_ACCURACY(   INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),   K2:K208 ) ` |
| L210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  L2:L208) ` |
| M210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  M2:M208) ` |
| N210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  N2:N208) ` |
| O210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  O2:O208) ` |
| P210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  P2:P208) ` |
| Q210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  Q2:Q208) ` |
| R210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  R2:R208) ` |
| S210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  S2:S208) ` |
| T210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  T2:T208) ` |
| U210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  U2:U208) ` |
| V210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  V2:V208) ` |
| W210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  W2:W208) ` |
| X210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  X2:X208) ` |
| Y210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  Y2:Y208) ` |
| Z210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  Z2:Z208) ` |
| AA210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AA2:AA208) ` |
| AB210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AB2:AB208) ` |
| AC210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AC2:AC208) ` |
| AD210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AD2:AD208) ` |
| AE210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AE2:AE208) ` |
| AF210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AF2:AF208) ` |
| AG210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AG2:AG208) ` |
| AH210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AH2:AH208) ` |
| AI210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AI2:AI208) ` |
| AJ210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AJ2:AJ208) ` |
| AK210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AK2:AK208) ` |
| AL210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AL2:AL208) ` |
| AM210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AM2:AM208) ` |
| AN210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AN2:AN208) ` |
| AO210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AO2:AO208) ` |
| AP210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AP2:AP208) ` |
| AQ210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AQ2:AQ208) ` |
| AR210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AR2:AR208) ` |
| AS210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AS2:AS208) ` |
| AT210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AT2:AT208) ` |
| AU210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AU2:AU208) ` |
| AV210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AV2:AV208) ` |
| AW210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AW2:AW208) ` |
| AX210 | `=MODEL_ACCURACY(   INDEX(2:208, , MATCH("FinalDiagnosisv2", 1:1, 0)),   AX2:AX208 ) ` |
| AY210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosisv2", 1:1, 0)),  AY2:AY208) ` |
| AZ210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosisv2", 1:1, 0)),  AZ2:AZ208) ` |
| BA210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosisv2", 1:1, 0)),  BA2:BA208) ` |
| BB210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosisv2", 1:1, 0)),  BB2:BB208) ` |
| BC210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosisv2", 1:1, 0)),  BC2:BC208) ` |
| BE210 | `=MODEL_ACCURACY(   INDEX(2:208, , MATCH("FinalDiagnosisv3_Omar", 1:1, 0)),   BE2:BE208 ) ` |
| BF210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosisv3_Omar", 1:1, 0)),  BF2:BF208) ` |
| BG210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosisv3_Omar", 1:1, 0)),  BG2:BG208) ` |
| BH210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosisv3_Omar", 1:1, 0)),  BH2:BH208) ` |
| BI210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosisv3_Omar", 1:1, 0)),  BI2:BI208) ` |
| BJ210 | `=MODEL_ACCURACY(   INDEX(2:208, , MATCH("FinalDiagnosisv2", 1:1, 0)),   BJ2:BJ208 ) ` |
| BK210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosisv2", 1:1, 0)),  BK2:BK208) ` |
| BL210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosisv2", 1:1, 0)),  BL2:BL208) ` |
| C211 | `=COUNTIF($B$2:$B$208, B211)` |
| C212 | `=COUNTIF($B$2:$B$208, B212)` |
| K212 | `=MCQ_CI_OPTIONB($F$2:$F$208, K2:K208, L2:L208, M2:M208) ` |
| C213 | `=COUNTIF($B$2:$B$208, B213)` |
| K213 | `=SC_ACCURACY_CI($F$2:$F$208, K2:K208, L2:L208, M2:M208)` |
| W213 | `=SC_ACCURACY_CI($F$2:$F$208, W2:W208, X2:X208, Y2:Y208)` |
| AC213 | `=SC_ACCURACY_CI($F$2:$F$208, AC2:AC208, AD2:AD208, AE2:AE208)` |
| AF213 | `=SC_ACCURACY_CI($F$2:$F$208, AF2:AF208, AG2:AG208, AH2:AH208)` |
| AI213 | `=SC_ACCURACY_CI($F$2:$F$208, AI2:AI208, AJ2:AJ208, AK2:AK208)` |
| AL213 | `=SC_ACCURACY_CI($F$2:$F$208, AL2:AL208, AM2:AM208, AN2:AN208)` |
| AO213 | `=SC_ACCURACY_CI($F$2:$F$208, AO2:AO208, AP2:AP208, AQ2:AQ208)` |
| AR213 | `=SC_ACCURACY_CI($F$2:$F$208, AR2:AR208, AS2:AS208, AT2:AT208)` |
| AU213 | `=SC_ACCURACY_CI($F$2:$F$208, AU2:AU208, AV2:AV208, AW2:AW208)` |
| C214 | `=COUNTIF($B$2:$B$208, B214)` |
| K214 | `=SC_ACCURACY_WILSON($F$2:$F$208, K2:K208, L2:L208, M2:M208)` |
| W214 | `=SC_ACCURACY_WILSON($F$2:$F$208, W2:W208, X2:X208, Y2:Y208)` |
| AC214 | `=SC_ACCURACY_WILSON($F$2:$F$208, AC2:AC208, AD2:AD208, AE2:AE208)` |
| AF214 | `=SC_ACCURACY_WILSON($F$2:$F$208, AF2:AF208, AG2:AG208, AH2:AH208)` |
| AI214 | `=SC_ACCURACY_WILSON($F$2:$F$208, AI2:AI208, AJ2:AJ208, AK2:AK208)` |
| AL214 | `=SC_ACCURACY_WILSON($F$2:$F$208, AL2:AL208, AM2:AM208, AN2:AN208)` |
| AO214 | `=SC_ACCURACY_WILSON($F$2:$F$208, AO2:AO208, AP2:AP208, AQ2:AQ208)` |
| AR214 | `=SC_ACCURACY_WILSON($F$2:$F$208, AR2:AR208, AS2:AS208, AT2:AT208)` |
| AU214 | `=SC_ACCURACY_WILSON($F$2:$F$208, AU2:AU208, AV2:AV208, AW2:AW208)` |
| C215 | `=COUNTIF($B$2:$B$208, B215)` |
| K215 | `=AVERAGE(K210:M210)` |
| W215 | `=AVERAGE(W210:Y210)` |
| AC215 | `=AVERAGE(AC210:AE210)` |
| AF215 | `=AVERAGE(AF210:AH210)` |
| AI215 | `=AVERAGE(AI210:AK210)` |
| AL215 | `=AVERAGE(AL210:AN210)` |
| AO215 | `=AVERAGE(AO210:AQ210)` |
| AR215 | `=AVERAGE(AR210:AT210)` |
| AU215 | `=AVERAGE(AU210:AW210)` |
| C216 | `=COUNTIF($B$2:$B$208, B216)` |
| C217 | `=COUNTIF($B$2:$B$208, B217)` |
| C218 | `=COUNTIF($B$2:$B$208, B218)` |
| C219 | `=COUNTIF($B$2:$B$208, B219)` |
| C220 | `=COUNTIF($B$2:$B$208, B220)` |
| K222 | `=MCQ_CI_TEXT($F$2:$F$208, K2:K208, L2:L208, M2:M208)` |
| W222 | `=MCQ_CI_TEXT($F$2:$F$208, W2:W208, X2:X208, Y2:Y208)` |
| AC222 | `=MCQ_CI_TEXT($F$2:$F$208, AC2:AC208, AD2:AD208, AE2:AE208)` |
| AF222 | `=MCQ_CI_TEXT($F$2:$F$208, AF2:AF208, AG2:AG208, AH2:AH208)` |
| AI222 | `=MCQ_CI_TEXT($F$2:$F$208, AI2:AI208, AJ2:AJ208, AK2:AK208)` |
| AL222 | `=MCQ_CI_TEXT($F$2:$F$208, AL2:AL208, AM2:AM208, AN2:AN208)` |
| AO222 | `=MCQ_CI_TEXT($F$2:$F$208, AO2:AO208, AP2:AP208, AQ2:AQ208)` |
| AR222 | `=MCQ_CI_TEXT($F$2:$F$208, AR2:AR208, AS2:AS208, AT2:AT208)` |
| AU222 | `=MCQ_CI_TEXT($F$2:$F$208, AU2:AU208, AV2:AV208, AW2:AW208)` |
| K223 | `=MCNEMAR_TEST(Eurorad!F2:F208, Eurorad!K2:M208, Eurorad!W2:Y208)` |
| K224 | `=MCNEMAR_VOTE_TEST(Eurorad!F2:F208, Eurorad!K2:M208, Eurorad!W2:Y208)` |
| W224 | `=MCNEMAR_VOTE_TEST(Eurorad!F2:F208, Eurorad!K2:M208, Eurorad!AC2:AE208)` |
| K225 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   Eurorad!F$2:F$208,   INDEX(     Eurorad!$2:$208,     ,     FILTER(       COLUMN(Eurorad!$1:$1),       REGEXMATCH(Eurorad!$1:$1, ""^gpt-5"")     )  ...` |
| K226 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   Eurorad!F2:F208,   TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!K2:Y208),       REGEXMATCH(TRANSPOSE(Eurorad!K1:Y1), ""^gpt-5"")     )   ),   TRA...` |
| W226 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   Eurorad!F2:F208,   TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!K2:AW208),       REGEXMATCH(TRANSPOSE(Eurorad!K1:Y1), ""^gpt-5"")     )   ),   TR...` |
| W227 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   Eurorad!F2:F208,   TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!K2:AW208),       REGEXMATCH(TRANSPOSE(Eurorad!K1:AW1), ""^gpt-5"")     )   ),   T...` |
| W228 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |
| W229 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Eurorad!$2:$208, , MATCH(""FinalDiagnosis"", Eurorad!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Eurorad!$2:$208),       REGEXMATCH...` |

### Omar Opthalmology (624 formulas)

| Cell | Formula |
|------|--------|
| C2 | `=CALC_ACCURACY("gpt5-0807-m", "Glaucoma", "Diagnosis")` |
| D2 | `=CALC_ACCURACY("o4-mini-m", "Glaucoma", "Diagnosis")` |
| E2 | `=CALC_ACCURACY("ds-r1-0528-v", "Glaucoma", "Diagnosis")` |
| F2 | `=CALC_ACCURACY("oss20b (L) v", "Glaucoma", "Diagnosis")` |
| G2 | `=CALC_ACCURACY("oss20b (M) v", "Glaucoma", "Diagnosis")` |
| H2 | `=CALC_ACCURACY("oss20b (H) v", "Glaucoma", "Diagnosis")` |
| I2 | `=CALC_ACCURACY("oss120b (L) v", "Glaucoma", "Diagnosis")` |
| J2 | `=CALC_ACCURACY("oss120b (M) v", "Glaucoma", "Diagnosis")` |
| K2 | `=CALC_ACCURACY("oss120b (H) v", "Glaucoma", "Diagnosis")` |
| C3 | `=CALC_ACCURACY("gpt5-0807-m", "External Eye/Orbital Diseases", "Diagnosis")` |
| D3 | `=CALC_ACCURACY("o4-mini-m", "External Eye/Orbital Diseases", "Diagnosis")` |
| E3 | `=CALC_ACCURACY("ds-r1-0528-v", "External Eye/Orbital Diseases", "Diagnosis")` |
| F3 | `=CALC_ACCURACY("oss20b (L) v", "External Eye/Orbital Diseases", "Diagnosis")` |
| G3 | `=CALC_ACCURACY("oss20b (M) v", "External Eye/Orbital Diseases", "Diagnosis")` |
| H3 | `=CALC_ACCURACY("oss20b (H) v", "External Eye/Orbital Diseases", "Diagnosis")` |
| I3 | `=CALC_ACCURACY("oss120b (L) v", "External Eye/Orbital Diseases", "Diagnosis")` |
| J3 | `=CALC_ACCURACY("oss120b (M) v", "External Eye/Orbital Diseases", "Diagnosis")` |
| K3 | `=CALC_ACCURACY("oss120b (H) v", "External Eye/Orbital Diseases", "Diagnosis")` |
| C4 | `=CALC_ACCURACY("gpt5-0807-m", "Retinal Diseases", "Diagnosis")` |
| D4 | `=CALC_ACCURACY("o4-mini-m", "Retinal Diseases", "Diagnosis")` |
| E4 | `=CALC_ACCURACY("ds-r1-0528-v", "Retinal Diseases", "Diagnosis")` |
| F4 | `=CALC_ACCURACY("oss20b (L) v", "Retinal Diseases", "Diagnosis")` |
| G4 | `=CALC_ACCURACY("oss20b (M) v", "Retinal Diseases", "Diagnosis")` |
| H4 | `=CALC_ACCURACY("oss20b (H) v", "Retinal Diseases", "Diagnosis")` |
| I4 | `=CALC_ACCURACY("oss120b (L) v", "Retinal Diseases", "Diagnosis")` |
| J4 | `=CALC_ACCURACY("oss120b (M) v", "Retinal Diseases", "Diagnosis")` |
| K4 | `=CALC_ACCURACY("oss120b (H) v", "Retinal Diseases", "Diagnosis")` |
| C5 | `=CALC_ACCURACY("gpt5-0807-m", "Anterior Segment Diseases", "Diagnosis")` |
| D5 | `=CALC_ACCURACY("o4-mini-m", "Anterior Segment Diseases", "Diagnosis")` |
| E5 | `=CALC_ACCURACY("ds-r1-0528-v", "Anterior Segment Diseases", "Diagnosis")` |
| F5 | `=CALC_ACCURACY("oss20b (L) v", "Anterior Segment Diseases", "Diagnosis")` |
| G5 | `=CALC_ACCURACY("oss20b (M) v", "Anterior Segment Diseases", "Diagnosis")` |
| H5 | `=CALC_ACCURACY("oss20b (H) v", "Anterior Segment Diseases", "Diagnosis")` |
| I5 | `=CALC_ACCURACY("oss120b (L) v", "Anterior Segment Diseases", "Diagnosis")` |
| J5 | `=CALC_ACCURACY("oss120b (M) v", "Anterior Segment Diseases", "Diagnosis")` |
| K5 | `=CALC_ACCURACY("oss120b (H) v", "Anterior Segment Diseases", "Diagnosis")` |
| C6 | `=CALC_ACCURACY("gpt5-0807-m", "Ocular Trauma", "Diagnosis")` |
| D6 | `=CALC_ACCURACY("o4-mini-m", "Ocular Trauma", "Diagnosis")` |
| E6 | `=CALC_ACCURACY("ds-r1-0528-v", "Ocular Trauma", "Diagnosis")` |
| F6 | `=CALC_ACCURACY("oss20b (L) v", "Ocular Trauma", "Diagnosis")` |
| G6 | `=CALC_ACCURACY("oss20b (M) v", "Ocular Trauma", "Diagnosis")` |
| H6 | `=CALC_ACCURACY("oss20b (H) v", "Ocular Trauma", "Diagnosis")` |
| I6 | `=CALC_ACCURACY("oss120b (L) v", "Ocular Trauma", "Diagnosis")` |
| J6 | `=CALC_ACCURACY("oss120b (M) v", "Ocular Trauma", "Diagnosis")` |
| K6 | `=CALC_ACCURACY("oss120b (H) v", "Ocular Trauma", "Diagnosis")` |
| C7 | `=CALC_ACCURACY("gpt5-0807-m", "Refractive Disorders/Strabismus", "Diagnosis")` |
| D7 | `=CALC_ACCURACY("o4-mini-m", "Refractive Disorders/Strabismus", "Diagnosis")` |
| E7 | `=CALC_ACCURACY("ds-r1-0528-v", "Refractive Disorders/Strabismus", "Diagnosis")` |
| F7 | `=CALC_ACCURACY("oss20b (L) v", "Refractive Disorders/Strabismus", "Diagnosis")` |
| G7 | `=CALC_ACCURACY("oss20b (M) v", "Refractive Disorders/Strabismus", "Diagnosis")` |
| H7 | `=CALC_ACCURACY("oss20b (H) v", "Refractive Disorders/Strabismus", "Diagnosis")` |
| I7 | `=CALC_ACCURACY("oss120b (L) v", "Refractive Disorders/Strabismus", "Diagnosis")` |
| J7 | `=CALC_ACCURACY("oss120b (M) v", "Refractive Disorders/Strabismus", "Diagnosis")` |
| K7 | `=CALC_ACCURACY("oss120b (H) v", "Refractive Disorders/Strabismus", "Diagnosis")` |
| C8 | `=CALC_ACCURACY("gpt5-0807-m", "Glaucoma", "Management")` |
| D8 | `=CALC_ACCURACY("o4-mini-m", "Glaucoma", "Management")` |
| E8 | `=CALC_ACCURACY("ds-r1-0528-v", "Glaucoma", "Management")` |
| F8 | `=CALC_ACCURACY("oss20b (L) v", "Glaucoma", "Management")` |
| G8 | `=CALC_ACCURACY("oss20b (M) v", "Glaucoma", "Management")` |
| H8 | `=CALC_ACCURACY("oss20b (H) v", "Glaucoma", "Management")` |
| I8 | `=CALC_ACCURACY("oss120b (L) v", "Glaucoma", "Management")` |
| J8 | `=CALC_ACCURACY("oss120b (M) v", "Glaucoma", "Management")` |
| K8 | `=CALC_ACCURACY("oss120b (H) v", "Glaucoma", "Management")` |
| C9 | `=CALC_ACCURACY("gpt5-0807-m", "External Eye/Orbital Diseases", "Management")` |
| D9 | `=CALC_ACCURACY("o4-mini-m", "External Eye/Orbital Diseases", "Management")` |
| E9 | `=CALC_ACCURACY("ds-r1-0528-v", "External Eye/Orbital Diseases", "Management")` |
| F9 | `=CALC_ACCURACY("oss20b (L) v", "External Eye/Orbital Diseases", "Management")` |
| G9 | `=CALC_ACCURACY("oss20b (M) v", "External Eye/Orbital Diseases", "Management")` |
| H9 | `=CALC_ACCURACY("oss20b (H) v", "External Eye/Orbital Diseases", "Management")` |
| I9 | `=CALC_ACCURACY("oss120b (L) v", "External Eye/Orbital Diseases", "Management")` |
| J9 | `=CALC_ACCURACY("oss120b (M) v", "External Eye/Orbital Diseases", "Management")` |
| K9 | `=CALC_ACCURACY("oss120b (H) v", "External Eye/Orbital Diseases", "Management")` |
| C10 | `=CALC_ACCURACY("gpt5-0807-m", "Retinal Diseases", "Management")` |
| D10 | `=CALC_ACCURACY("o4-mini-m", "Retinal Diseases", "Management")` |
| E10 | `=CALC_ACCURACY("ds-r1-0528-v", "Retinal Diseases", "Management")` |
| F10 | `=CALC_ACCURACY("oss20b (L) v", "Retinal Diseases", "Management")` |
| G10 | `=CALC_ACCURACY("oss20b (M) v", "Retinal Diseases", "Management")` |
| H10 | `=CALC_ACCURACY("oss20b (H) v", "Retinal Diseases", "Management")` |
| I10 | `=CALC_ACCURACY("oss120b (L) v", "Retinal Diseases", "Management")` |
| J10 | `=CALC_ACCURACY("oss120b (M) v", "Retinal Diseases", "Management")` |
| K10 | `=CALC_ACCURACY("oss120b (H) v", "Retinal Diseases", "Management")` |
| C11 | `=CALC_ACCURACY("gpt5-0807-m", "Anterior Segment Diseases", "Management")` |
| D11 | `=CALC_ACCURACY("o4-mini-m", "Anterior Segment Diseases", "Management")` |
| E11 | `=CALC_ACCURACY("ds-r1-0528-v", "Anterior Segment Diseases", "Management")` |
| F11 | `=CALC_ACCURACY("oss20b (L) v", "Anterior Segment Diseases", "Management")` |
| G11 | `=CALC_ACCURACY("oss20b (M) v", "Anterior Segment Diseases", "Management")` |
| H11 | `=CALC_ACCURACY("oss20b (H) v", "Anterior Segment Diseases", "Management")` |
| I11 | `=CALC_ACCURACY("oss120b (L) v", "Anterior Segment Diseases", "Management")` |
| J11 | `=CALC_ACCURACY("oss120b (M) v", "Anterior Segment Diseases", "Management")` |
| K11 | `=CALC_ACCURACY("oss120b (H) v", "Anterior Segment Diseases", "Management")` |
| C12 | `=CALC_ACCURACY("gpt5-0807-m", "Ocular Trauma", "Management")` |
| D12 | `=CALC_ACCURACY("o4-mini-m", "Ocular Trauma", "Management")` |
| E12 | `=CALC_ACCURACY("ds-r1-0528-v", "Ocular Trauma", "Management")` |
| F12 | `=CALC_ACCURACY("oss20b (L) v", "Ocular Trauma", "Management")` |
| G12 | `=CALC_ACCURACY("oss20b (M) v", "Ocular Trauma", "Management")` |
| H12 | `=CALC_ACCURACY("oss20b (H) v", "Ocular Trauma", "Management")` |
| I12 | `=CALC_ACCURACY("oss120b (L) v", "Ocular Trauma", "Management")` |
| J12 | `=CALC_ACCURACY("oss120b (M) v", "Ocular Trauma", "Management")` |
| K12 | `=CALC_ACCURACY("oss120b (H) v", "Ocular Trauma", "Management")` |
| C13 | `=CALC_ACCURACY("gpt5-0807-m", "Refractive Disorders/Strabismus", "Management")` |
| D13 | `=CALC_ACCURACY("o4-mini-m", "Refractive Disorders/Strabismus", "Management")` |
| E13 | `=CALC_ACCURACY("ds-r1-0528-v", "Refractive Disorders/Strabismus", "Management")` |
| F13 | `=CALC_ACCURACY("oss20b (L) v", "Refractive Disorders/Strabismus", "Management")` |
| G13 | `=CALC_ACCURACY("oss20b (M) v", "Refractive Disorders/Strabismus", "Management")` |
| H13 | `=CALC_ACCURACY("oss20b (H) v", "Refractive Disorders/Strabismus", "Management")` |
| I13 | `=CALC_ACCURACY("oss120b (L) v", "Refractive Disorders/Strabismus", "Management")` |
| J13 | `=CALC_ACCURACY("oss120b (M) v", "Refractive Disorders/Strabismus", "Management")` |
| K13 | `=CALC_ACCURACY("oss120b (H) v", "Refractive Disorders/Strabismus", "Management")` |
| C14 | `=CALC_ACCURACY("gpt5-0807-m", "", "Diagnosis")` |
| D14 | `=CALC_ACCURACY("o4-mini-m", "", "Diagnosis")` |
| E14 | `=CALC_ACCURACY("ds-r1-0528-v", "", "Diagnosis")` |
| F14 | `=CALC_ACCURACY("oss20b (L) v", "", "Diagnosis")` |
| G14 | `=CALC_ACCURACY("oss20b (M) v", "", "Diagnosis")` |
| H14 | `=CALC_ACCURACY("oss20b (H) v", "", "Diagnosis")` |
| I14 | `=CALC_ACCURACY("oss120b (L) v", "", "Diagnosis")` |
| J14 | `=CALC_ACCURACY("oss120b (M) v", "", "Diagnosis")` |
| K14 | `=CALC_ACCURACY("oss120b (H) v", "", "Diagnosis")` |
| C15 | `=CALC_ACCURACY("gpt5-0807-m", "", "Management")` |
| D15 | `=CALC_ACCURACY("o4-mini-m", "", "Management")` |
| E15 | `=CALC_ACCURACY("ds-r1-0528-v", "", "Management")` |
| F15 | `=CALC_ACCURACY("oss20b (L) v", "", "Management")` |
| G15 | `=CALC_ACCURACY("oss20b (M) v", "", "Management")` |
| H15 | `=CALC_ACCURACY("oss20b (H) v", "", "Management")` |
| I15 | `=CALC_ACCURACY("oss120b (L) v", "", "Management")` |
| J15 | `=CALC_ACCURACY("oss120b (M) v", "", "Management")` |
| K15 | `=CALC_ACCURACY("oss120b (H) v", "", "Management")` |
| C16 | `=COMBINE_DIAG_MANAG(C14, C15, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| D16 | `=COMBINE_DIAG_MANAG(D14, D15, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| E16 | `=COMBINE_DIAG_MANAG(E14, E15, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| F16 | `=COMBINE_DIAG_MANAG(F14, F15, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| G16 | `=COMBINE_DIAG_MANAG(G14, G15, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| H16 | `=COMBINE_DIAG_MANAG(H14, H15, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| I16 | `=COMBINE_DIAG_MANAG(I14, I15, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| J16 | `=COMBINE_DIAG_MANAG(J14, J15, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| K16 | `=COMBINE_DIAG_MANAG(K14, K15, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| C17 | `=CALC_ACCURACY("gpt5-0807-m", "", "")` |
| D17 | `=CALC_ACCURACY("o4-mini-m", "", "")` |
| E17 | `=CALC_ACCURACY("ds-r1-0528-v", "", "")` |
| F17 | `=CALC_ACCURACY("oss20b (L) v", "", "")` |
| G17 | `=CALC_ACCURACY("oss20b (M) v", "", "")` |
| H17 | `=CALC_ACCURACY("oss20b (H) v", "", "")` |
| I17 | `=CALC_ACCURACY("oss120b (L) v", "", "")` |
| J17 | `=CALC_ACCURACY("oss120b (M) v", "", "")` |
| K17 | `=CALC_ACCURACY("oss120b (H) v", "", "")` |
| C29 | `=CALC_ACCURACY_UPDATED(C26, B29,A29)` |
| D29 | `=CALC_ACCURACY_UPDATED(D26,B29,A29)` |
| E29 | `=CALC_ACCURACY_UPDATED(E26, B29, A29)` |
| F29 | `=CALC_ACCURACY_UPDATED(F26, B29,A29)` |
| G29 | `=CALC_ACCURACY_UPDATED(G26, B29,A29)` |
| H29 | `=CALC_ACCURACY_UPDATED(H26, B29,A29)` |
| I29 | `=CALC_ACCURACY_UPDATED(I26, B29,A29)` |
| J29 | `=CALC_ACCURACY_UPDATED(J26, B29,A29)` |
| K29 | `=CALC_ACCURACY_UPDATED(K26, B29,A29)` |
| L29 | `=CALC_ACCURACY_UPDATED(L26, B29, A29)` |
| M29 | `=CALC_ACCURACY_UPDATED(M26, B29, A29)` |
| N29 | `=CALC_ACCURACY_UPDATED(N26, B29, A29)` |
| O29 | `=CALC_ACCURACY_UPDATED(O26, B29, A29)` |
| P29 | `=CALC_ACCURACY_UPDATED(P26, B29, A29)` |
| Q29 | `=CALC_ACCURACY_UPDATED(Q26, B29, A29)` |
| C30 | `=CALC_ACCURACY_UPDATED(C26, B30,A30)` |
| D30 | `=CALC_ACCURACY_UPDATED(D26,B30,A30)` |
| E30 | `=CALC_ACCURACY_UPDATED(E26, B30, A30)` |
| F30 | `=CALC_ACCURACY_UPDATED(F26, B30,A30)` |
| G30 | `=CALC_ACCURACY_UPDATED(G26, B30,A30)` |
| H30 | `=CALC_ACCURACY_UPDATED(H26, B30,A30)` |
| I30 | `=CALC_ACCURACY_UPDATED(I26, B30,A30)` |
| J30 | `=CALC_ACCURACY_UPDATED(J26, B30,A30)` |
| K30 | `=CALC_ACCURACY_UPDATED(K26, B30,A30)` |
| L30 | `=CALC_ACCURACY_UPDATED(L26, B30, A30)` |
| M30 | `=CALC_ACCURACY_UPDATED(M26, B30, A30)` |
| N30 | `=CALC_ACCURACY_UPDATED(N26, B30, A30)` |
| O30 | `=CALC_ACCURACY_UPDATED(O26, B30, A30)` |
| P30 | `=CALC_ACCURACY_UPDATED(P26, B30, A30)` |
| Q30 | `=CALC_ACCURACY_UPDATED(Q26, B30, A30)` |
| C31 | `=CALC_ACCURACY_UPDATED(C26, B31,A31)` |
| D31 | `=CALC_ACCURACY_UPDATED(D26,B31,A31)` |
| E31 | `=CALC_ACCURACY_UPDATED(E26, B31, A31)` |
| F31 | `=CALC_ACCURACY_UPDATED(F26, B31,A31)` |
| G31 | `=CALC_ACCURACY_UPDATED(G26, B31,A31)` |
| H31 | `=CALC_ACCURACY_UPDATED(H26, B31,A31)` |
| I31 | `=CALC_ACCURACY_UPDATED(I26, B31,A31)` |
| J31 | `=CALC_ACCURACY_UPDATED(J26, B31,A31)` |
| K31 | `=CALC_ACCURACY_UPDATED(K26, B31,A31)` |
| L31 | `=CALC_ACCURACY_UPDATED(L26, B31, A31)` |
| M31 | `=CALC_ACCURACY_UPDATED(M26, B31, A31)` |
| N31 | `=CALC_ACCURACY_UPDATED(N26, B31, A31)` |
| O31 | `=CALC_ACCURACY_UPDATED(O26, B31, A31)` |
| P31 | `=CALC_ACCURACY_UPDATED(P26, B31, A31)` |
| Q31 | `=CALC_ACCURACY_UPDATED(Q26, B31, A31)` |
| C32 | `=CALC_ACCURACY_UPDATED(C26, B32,A32)` |
| D32 | `=CALC_ACCURACY_UPDATED(D26,B32,A32)` |
| E32 | `=CALC_ACCURACY_UPDATED(E26, B32, A32)` |
| F32 | `=CALC_ACCURACY_UPDATED(F26, B32,A32)` |
| G32 | `=CALC_ACCURACY_UPDATED(G26, B32,A32)` |
| H32 | `=CALC_ACCURACY_UPDATED(H26, B32,A32)` |
| I32 | `=CALC_ACCURACY_UPDATED(I26, B32,A32)` |
| J32 | `=CALC_ACCURACY_UPDATED(J26, B32,A32)` |
| K32 | `=CALC_ACCURACY_UPDATED(K26, B32,A32)` |
| L32 | `=CALC_ACCURACY_UPDATED(L26, B32, A32)` |
| M32 | `=CALC_ACCURACY_UPDATED(M26, B32, A32)` |
| N32 | `=CALC_ACCURACY_UPDATED(N26, B32, A32)` |
| O32 | `=CALC_ACCURACY_UPDATED(O26, B32, A32)` |
| P32 | `=CALC_ACCURACY_UPDATED(P26, B32, A32)` |
| Q32 | `=CALC_ACCURACY_UPDATED(Q26, B32, A32)` |
| C33 | `=CALC_ACCURACY_UPDATED(C26, B33,A33)` |
| D33 | `=CALC_ACCURACY_UPDATED(D26,B33,A33)` |
| E33 | `=CALC_ACCURACY_UPDATED(E26, B33, A33)` |
| F33 | `=CALC_ACCURACY_UPDATED(F26, B33,A33)` |
| G33 | `=CALC_ACCURACY_UPDATED(G26, B33,A33)` |
| H33 | `=CALC_ACCURACY_UPDATED(H26, B33,A33)` |
| I33 | `=CALC_ACCURACY_UPDATED(I26, B33,A33)` |
| J33 | `=CALC_ACCURACY_UPDATED(J26, B33,A33)` |
| K33 | `=CALC_ACCURACY_UPDATED(K26, B33,A33)` |
| L33 | `=CALC_ACCURACY_UPDATED(L26, B33, A33)` |
| M33 | `=CALC_ACCURACY_UPDATED(M26, B33, A33)` |
| N33 | `=CALC_ACCURACY_UPDATED(N26, B33, A33)` |
| O33 | `=CALC_ACCURACY_UPDATED(O26, B33, A33)` |
| P33 | `=CALC_ACCURACY_UPDATED(P26, B33, A33)` |
| Q33 | `=CALC_ACCURACY_UPDATED(Q26, B33, A33)` |
| C34 | `=CALC_ACCURACY_UPDATED(C26, B34,A34)` |
| D34 | `=CALC_ACCURACY_UPDATED(D26,B34,A34)` |
| E34 | `=CALC_ACCURACY_UPDATED(E26, B34, A34)` |
| F34 | `=CALC_ACCURACY_UPDATED(F26, B34,A34)` |
| G34 | `=CALC_ACCURACY_UPDATED(G26, B34,A34)` |
| H34 | `=CALC_ACCURACY_UPDATED(H26, B34,A34)` |
| I34 | `=CALC_ACCURACY_UPDATED(I26, B34,A34)` |
| J34 | `=CALC_ACCURACY_UPDATED(J26, B34,A34)` |
| K34 | `=CALC_ACCURACY_UPDATED(K26, B34,A34)` |
| L34 | `=CALC_ACCURACY_UPDATED(L26, B34, A34)` |
| M34 | `=CALC_ACCURACY_UPDATED(M26, B34, A34)` |
| N34 | `=CALC_ACCURACY_UPDATED(N26, B34, A34)` |
| O34 | `=CALC_ACCURACY_UPDATED(O26, B34, A34)` |
| P34 | `=CALC_ACCURACY_UPDATED(P26, B34, A34)` |
| Q34 | `=CALC_ACCURACY_UPDATED(Q26, B34, A34)` |
| C35 | `=CALC_ACCURACY_UPDATED(C26, B35,A35)` |
| D35 | `=CALC_ACCURACY_UPDATED(D26,B35,A35)` |
| E35 | `=CALC_ACCURACY_UPDATED(E26, B35, A35)` |
| F35 | `=CALC_ACCURACY_UPDATED(F26, B35,A35)` |
| G35 | `=CALC_ACCURACY_UPDATED(G26, B35,A35)` |
| H35 | `=CALC_ACCURACY_UPDATED(H26, B35,A35)` |
| I35 | `=CALC_ACCURACY_UPDATED(I26, B35,A35)` |
| J35 | `=CALC_ACCURACY_UPDATED(J26, B35,A35)` |
| K35 | `=CALC_ACCURACY_UPDATED(K26, B35,A35)` |
| L35 | `=CALC_ACCURACY_UPDATED(L26, B35, A35)` |
| M35 | `=CALC_ACCURACY_UPDATED(M26, B35, A35)` |
| N35 | `=CALC_ACCURACY_UPDATED(N26, B35, A35)` |
| O35 | `=CALC_ACCURACY_UPDATED(O26, B35, A35)` |
| P35 | `=CALC_ACCURACY_UPDATED(P26, B35, A35)` |
| Q35 | `=CALC_ACCURACY_UPDATED(Q26, B35, A35)` |
| C36 | `=CALC_ACCURACY_UPDATED(C26, B36,A36)` |
| D36 | `=CALC_ACCURACY_UPDATED(D26,B36,A36)` |
| E36 | `=CALC_ACCURACY_UPDATED(E26, B36, A36)` |
| F36 | `=CALC_ACCURACY_UPDATED(F26, B36,A36)` |
| G36 | `=CALC_ACCURACY_UPDATED(G26, B36,A36)` |
| H36 | `=CALC_ACCURACY_UPDATED(H26, B36,A36)` |
| I36 | `=CALC_ACCURACY_UPDATED(I26, B36,A36)` |
| J36 | `=CALC_ACCURACY_UPDATED(J26, B36,A36)` |
| K36 | `=CALC_ACCURACY_UPDATED(K26, B36,A36)` |
| L36 | `=CALC_ACCURACY_UPDATED(L26, B36, A36)` |
| M36 | `=CALC_ACCURACY_UPDATED(M26, B36, A36)` |
| N36 | `=CALC_ACCURACY_UPDATED(N26, B36, A36)` |
| O36 | `=CALC_ACCURACY_UPDATED(O26, B36, A36)` |
| P36 | `=CALC_ACCURACY_UPDATED(P26, B36, A36)` |
| Q36 | `=CALC_ACCURACY_UPDATED(Q26, B36, A36)` |
| C37 | `=CALC_ACCURACY_UPDATED(C26, B37,A37)` |
| D37 | `=CALC_ACCURACY_UPDATED(D26,B37,A37)` |
| E37 | `=CALC_ACCURACY_UPDATED(E26, B37, A37)` |
| F37 | `=CALC_ACCURACY_UPDATED(F26, B37,A37)` |
| G37 | `=CALC_ACCURACY_UPDATED(G26, B37,A37)` |
| H37 | `=CALC_ACCURACY_UPDATED(H26, B37,A37)` |
| I37 | `=CALC_ACCURACY_UPDATED(I26, B37,A37)` |
| J37 | `=CALC_ACCURACY_UPDATED(J26, B37,A37)` |
| K37 | `=CALC_ACCURACY_UPDATED(K26, B37,A37)` |
| L37 | `=CALC_ACCURACY_UPDATED(L26, B37, A37)` |
| M37 | `=CALC_ACCURACY_UPDATED(M26, B37, A37)` |
| N37 | `=CALC_ACCURACY_UPDATED(N26, B37, A37)` |
| O37 | `=CALC_ACCURACY_UPDATED(O26, B37, A37)` |
| P37 | `=CALC_ACCURACY_UPDATED(P26, B37, A37)` |
| Q37 | `=CALC_ACCURACY_UPDATED(Q26, B37, A37)` |
| C38 | `=CALC_ACCURACY_UPDATED(C26, B38,A38)` |
| D38 | `=CALC_ACCURACY_UPDATED(D26,B38,A38)` |
| E38 | `=CALC_ACCURACY_UPDATED(E26, B38, A38)` |
| F38 | `=CALC_ACCURACY_UPDATED(F26, B38,A38)` |
| G38 | `=CALC_ACCURACY_UPDATED(G26, B38,A38)` |
| H38 | `=CALC_ACCURACY_UPDATED(H26, B38,A38)` |
| I38 | `=CALC_ACCURACY_UPDATED(I26, B38,A38)` |
| J38 | `=CALC_ACCURACY_UPDATED(J26, B38,A38)` |
| K38 | `=CALC_ACCURACY_UPDATED(K26, B38,A38)` |
| L38 | `=CALC_ACCURACY_UPDATED(L26, B38, A38)` |
| M38 | `=CALC_ACCURACY_UPDATED(M26, B38, A38)` |
| N38 | `=CALC_ACCURACY_UPDATED(N26, B38, A38)` |
| O38 | `=CALC_ACCURACY_UPDATED(O26, B38, A38)` |
| P38 | `=CALC_ACCURACY_UPDATED(P26, B38, A38)` |
| Q38 | `=CALC_ACCURACY_UPDATED(Q26, B38, A38)` |
| C39 | `=CALC_ACCURACY_UPDATED(C26, B39,A39)` |
| D39 | `=CALC_ACCURACY_UPDATED(D26,B39,A39)` |
| E39 | `=CALC_ACCURACY_UPDATED(E26, B39, A39)` |
| F39 | `=CALC_ACCURACY_UPDATED(F26, B39,A39)` |
| G39 | `=CALC_ACCURACY_UPDATED(G26, B39,A39)` |
| H39 | `=CALC_ACCURACY_UPDATED(H26, B39,A39)` |
| I39 | `=CALC_ACCURACY_UPDATED(I26, B39,A39)` |
| J39 | `=CALC_ACCURACY_UPDATED(J26, B39,A39)` |
| K39 | `=CALC_ACCURACY_UPDATED(K26, B39,A39)` |
| L39 | `=CALC_ACCURACY_UPDATED(L26, B39, A39)` |
| M39 | `=CALC_ACCURACY_UPDATED(M26, B39, A39)` |
| N39 | `=CALC_ACCURACY_UPDATED(N26, B39, A39)` |
| O39 | `=CALC_ACCURACY_UPDATED(O26, B39, A39)` |
| P39 | `=CALC_ACCURACY_UPDATED(P26, B39, A39)` |
| Q39 | `=CALC_ACCURACY_UPDATED(Q26, B39, A39)` |
| C40 | `=CALC_ACCURACY_UPDATED(C26,B40,A40)` |
| D40 | `=CALC_ACCURACY_UPDATED(D26,B40,A40)` |
| E40 | `=CALC_ACCURACY_UPDATED(E26, B40, A40)` |
| F40 | `=CALC_ACCURACY_UPDATED(F26, B40,A40)` |
| G40 | `=CALC_ACCURACY_UPDATED(G26, B40,A40)` |
| H40 | `=CALC_ACCURACY_UPDATED(H26, B40,A40)` |
| I40 | `=CALC_ACCURACY_UPDATED(I26, B40,A40)` |
| J40 | `=CALC_ACCURACY_UPDATED(J26, B40,A40)` |
| K40 | `=CALC_ACCURACY_UPDATED(K26, B40,A40)` |
| L40 | `=CALC_ACCURACY_UPDATED(L26, B40, A40)` |
| M40 | `=CALC_ACCURACY_UPDATED(M26, B40, A40)` |
| N40 | `=CALC_ACCURACY_UPDATED(N26, B40, A40)` |
| O40 | `=CALC_ACCURACY_UPDATED(O26, B40, A40)` |
| P40 | `=CALC_ACCURACY_UPDATED(P26, B40, A40)` |
| Q40 | `=CALC_ACCURACY_UPDATED(Q26, B40, A40)` |
| C41 | `=CALC_ACCURACY_UPDATED(C26, "", "Diagnosis")` |
| D41 | `=CALC_ACCURACY(D26, "", "Diagnosis")` |
| E41 | `=CALC_ACCURACY_UPDATED(E26, B41, "Diagnosis")` |
| F41 | `=CALC_ACCURACY_UPDATED(F26, B41, "Diagnosis")` |
| G41 | `=CALC_ACCURACY_UPDATED(G26, B41, "Diagnosis")` |
| H41 | `=CALC_ACCURACY_UPDATED(H26, B41, "Diagnosis")` |
| I41 | `=CALC_ACCURACY_UPDATED(I26, B41, "Diagnosis")` |
| J41 | `=CALC_ACCURACY_UPDATED(J26, B41, "Diagnosis")` |
| K41 | `=CALC_ACCURACY_UPDATED(K26, B41, "Diagnosis")` |
| L41 | `=CALC_ACCURACY_UPDATED(L26, B41, "Diagnosis")` |
| M41 | `=CALC_ACCURACY_UPDATED(M26, B41, "Diagnosis")` |
| N41 | `=CALC_ACCURACY_UPDATED(N26, B41, "Diagnosis")` |
| O41 | `=CALC_ACCURACY_UPDATED(O26, B41, "Diagnosis")` |
| P41 | `=CALC_ACCURACY_UPDATED(P26, B41, "Diagnosis")` |
| Q41 | `=CALC_ACCURACY_UPDATED(Q26, B41, "Diagnosis")` |
| C42 | `=CALC_ACCURACY_UPDATED(C26, "", "Management")` |
| D42 | `=CALC_ACCURACY_UPDATED(D26, "", "Management")` |
| E42 | `=CALC_ACCURACY_UPDATED(E26, B42, "Management")` |
| F42 | `=CALC_ACCURACY_UPDATED(F26, B42, "Management")` |
| G42 | `=CALC_ACCURACY_UPDATED(G26, B42, "Management")` |
| H42 | `=CALC_ACCURACY_UPDATED(H26, B42, "Management")` |
| I42 | `=CALC_ACCURACY_UPDATED(I26, B42, "Management")` |
| J42 | `=CALC_ACCURACY_UPDATED(J26, B42, "Management")` |
| K42 | `=CALC_ACCURACY_UPDATED(K26, B42, "Management")` |
| L42 | `=CALC_ACCURACY_UPDATED(L26, B42, "Management")` |
| M42 | `=CALC_ACCURACY_UPDATED(M26, B42, "Management")` |
| N42 | `=CALC_ACCURACY_UPDATED(N26, B42, "Management")` |
| O42 | `=CALC_ACCURACY_UPDATED(O26, B42, "Management")` |
| P42 | `=CALC_ACCURACY_UPDATED(P26, B42, "Management")` |
| Q42 | `=CALC_ACCURACY_UPDATED(Q26, B42, "Management")` |
| C43 | `=COMBINE_DIAG_MANAG(C41, C42, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| D43 | `=COMBINE_DIAG_MANAG(D41, D42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| E43 | `=COMBINE_DIAG_MANAG(E41, E42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| F43 | `=COMBINE_DIAG_MANAG(F41, F42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| G43 | `=COMBINE_DIAG_MANAG(G41, G42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| H43 | `=COMBINE_DIAG_MANAG(H41, H42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| I43 | `=COMBINE_DIAG_MANAG(I41, I42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| J43 | `=COMBINE_DIAG_MANAG(J41, J42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| K43 | `=COMBINE_DIAG_MANAG(K41, K42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| L43 | `=COMBINE_DIAG_MANAG(L41, L42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| M43 | `=COMBINE_DIAG_MANAG(M41, M42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| N43 | `=COMBINE_DIAG_MANAG(N41, N42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| O43 | `=COMBINE_DIAG_MANAG(O41, O42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| P43 | `=COMBINE_DIAG_MANAG(P41, P42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| Q43 | `=COMBINE_DIAG_MANAG(Q41, Q42, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| C44 | `=CALC_ACCURACY_UPDATED(C26, "", "")` |
| D44 | `=CALC_ACCURACY_UPDATED(D26, "", "")` |
| E44 | `=CALC_ACCURACY_UPDATED(E26, "", "")` |
| F44 | `=CALC_ACCURACY_UPDATED(F26, "", "")` |
| G44 | `=CALC_ACCURACY_UPDATED(G26, "", "")` |
| H44 | `=CALC_ACCURACY_UPDATED(H26, "", "")` |
| I44 | `=CALC_ACCURACY_UPDATED(I26, "", "")` |
| J44 | `=CALC_ACCURACY_UPDATED(J26, "", "")` |
| K44 | `=CALC_ACCURACY_UPDATED(K26, "", "")` |
| L44 | `=CALC_ACCURACY_UPDATED(L26, "", "")` |
| M44 | `=CALC_ACCURACY_UPDATED(M26, "", "")` |
| N44 | `=CALC_ACCURACY_UPDATED(N26, "", "")` |
| O44 | `=CALC_ACCURACY_UPDATED(O26, "", "")` |
| P44 | `=CALC_ACCURACY_UPDATED(P26, "", "")` |
| Q44 | `=CALC_ACCURACY_UPDATED(Q26, "", "")` |
| C67 | `=CALC_ACCURACY_UPDATED(C64, B67,A67)` |
| D67 | `=CALC_ACCURACY_UPDATED(D64,B67,A67)` |
| E67 | `=CALC_ACCURACY_UPDATED(E64, B67, A67)` |
| F67 | `=CALC_ACCURACY_UPDATED(F64, B67,A67)` |
| G67 | `=CALC_ACCURACY_UPDATED(G64, B67,A67)` |
| H67 | `=CALC_ACCURACY_UPDATED(H64, B67,A67)` |
| I67 | `=CALC_ACCURACY_UPDATED(I64, B67,A67)` |
| J67 | `=CALC_ACCURACY_UPDATED(J64, B67,A67)` |
| K67 | `=CALC_ACCURACY_UPDATED(K64, B67,A67)` |
| L67 | `=CALC_ACCURACY_UPDATED(L64, B67, A67)` |
| M67 | `=CALC_ACCURACY_UPDATED(M64, B67, A67)` |
| N67 | `=CALC_ACCURACY_UPDATED(N64, B67, A67)` |
| O67 | `=CALC_ACCURACY_UPDATED(O64, B67, A67)` |
| P67 | `=CALC_ACCURACY_UPDATED(P64, B67, A67)` |
| Q67 | `=CALC_ACCURACY_UPDATED(Q64, B67, A67)` |
| C68 | `=CALC_ACCURACY_UPDATED(C64, B68,A68)` |
| D68 | `=CALC_ACCURACY_UPDATED(D64,B68,A68)` |
| E68 | `=CALC_ACCURACY_UPDATED(E64, B68, A68)` |
| F68 | `=CALC_ACCURACY_UPDATED(F64, B68,A68)` |
| G68 | `=CALC_ACCURACY_UPDATED(G64, B68,A68)` |
| H68 | `=CALC_ACCURACY_UPDATED(H64, B68,A68)` |
| I68 | `=CALC_ACCURACY_UPDATED(I64, B68,A68)` |
| J68 | `=CALC_ACCURACY_UPDATED(J64, B68,A68)` |
| K68 | `=CALC_ACCURACY_UPDATED(K64, B68,A68)` |
| L68 | `=CALC_ACCURACY_UPDATED(L64, B68, A68)` |
| M68 | `=CALC_ACCURACY_UPDATED(M64, B68, A68)` |
| N68 | `=CALC_ACCURACY_UPDATED(N64, B68, A68)` |
| O68 | `=CALC_ACCURACY_UPDATED(O64, B68, A68)` |
| P68 | `=CALC_ACCURACY_UPDATED(P64, B68, A68)` |
| Q68 | `=CALC_ACCURACY_UPDATED(Q64, B68, A68)` |
| C69 | `=CALC_ACCURACY_UPDATED(C64, B69,A69)` |
| D69 | `=CALC_ACCURACY_UPDATED(D64,B69,A69)` |
| E69 | `=CALC_ACCURACY_UPDATED(E64, B69, A69)` |
| F69 | `=CALC_ACCURACY_UPDATED(F64, B69,A69)` |
| G69 | `=CALC_ACCURACY_UPDATED(G64, B69,A69)` |
| H69 | `=CALC_ACCURACY_UPDATED(H64, B69,A69)` |
| I69 | `=CALC_ACCURACY_UPDATED(I64, B69,A69)` |
| J69 | `=CALC_ACCURACY_UPDATED(J64, B69,A69)` |
| K69 | `=CALC_ACCURACY_UPDATED(K64, B69,A69)` |
| L69 | `=CALC_ACCURACY_UPDATED(L64, B69, A69)` |
| M69 | `=CALC_ACCURACY_UPDATED(M64, B69, A69)` |
| N69 | `=CALC_ACCURACY_UPDATED(N64, B69, A69)` |
| O69 | `=CALC_ACCURACY_UPDATED(O64, B69, A69)` |
| P69 | `=CALC_ACCURACY_UPDATED(P64, B69, A69)` |
| Q69 | `=CALC_ACCURACY_UPDATED(Q64, B69, A69)` |
| C70 | `=CALC_ACCURACY_UPDATED(C64, B70,A70)` |
| D70 | `=CALC_ACCURACY_UPDATED(D64,B70,A70)` |
| E70 | `=CALC_ACCURACY_UPDATED(E64, B70, A70)` |
| F70 | `=CALC_ACCURACY_UPDATED(F64, B70,A70)` |
| G70 | `=CALC_ACCURACY_UPDATED(G64, B70,A70)` |
| H70 | `=CALC_ACCURACY_UPDATED(H64, B70,A70)` |
| I70 | `=CALC_ACCURACY_UPDATED(I64, B70,A70)` |
| J70 | `=CALC_ACCURACY_UPDATED(J64, B70,A70)` |
| K70 | `=CALC_ACCURACY_UPDATED(K64, B70,A70)` |
| L70 | `=CALC_ACCURACY_UPDATED(L64, B70, A70)` |
| M70 | `=CALC_ACCURACY_UPDATED(M64, B70, A70)` |
| N70 | `=CALC_ACCURACY_UPDATED(N64, B70, A70)` |
| O70 | `=CALC_ACCURACY_UPDATED(O64, B70, A70)` |
| P70 | `=CALC_ACCURACY_UPDATED(P64, B70, A70)` |
| Q70 | `=CALC_ACCURACY_UPDATED(Q64, B70, A70)` |
| C71 | `=CALC_ACCURACY_UPDATED(C64, B71,A71)` |
| D71 | `=CALC_ACCURACY_UPDATED(D64,B71,A71)` |
| E71 | `=CALC_ACCURACY_UPDATED(E64, B71, A71)` |
| F71 | `=CALC_ACCURACY_UPDATED(F64, B71,A71)` |
| G71 | `=CALC_ACCURACY_UPDATED(G64, B71,A71)` |
| H71 | `=CALC_ACCURACY_UPDATED(H64, B71,A71)` |
| I71 | `=CALC_ACCURACY_UPDATED(I64, B71,A71)` |
| J71 | `=CALC_ACCURACY_UPDATED(J64, B71,A71)` |
| K71 | `=CALC_ACCURACY_UPDATED(K64, B71,A71)` |
| L71 | `=CALC_ACCURACY_UPDATED(L64, B71, A71)` |
| M71 | `=CALC_ACCURACY_UPDATED(M64, B71, A71)` |
| N71 | `=CALC_ACCURACY_UPDATED(N64, B71, A71)` |
| O71 | `=CALC_ACCURACY_UPDATED(O64, B71, A71)` |
| P71 | `=CALC_ACCURACY_UPDATED(P64, B71, A71)` |
| Q71 | `=CALC_ACCURACY_UPDATED(Q64, B71, A71)` |
| C72 | `=CALC_ACCURACY_UPDATED(C64, B72,A72)` |
| D72 | `=CALC_ACCURACY_UPDATED(D64,B72,A72)` |
| E72 | `=CALC_ACCURACY_UPDATED(E64, B72, A72)` |
| F72 | `=CALC_ACCURACY_UPDATED(F64, B72,A72)` |
| G72 | `=CALC_ACCURACY_UPDATED(G64, B72,A72)` |
| H72 | `=CALC_ACCURACY_UPDATED(H64, B72,A72)` |
| I72 | `=CALC_ACCURACY_UPDATED(I64, B72,A72)` |
| J72 | `=CALC_ACCURACY_UPDATED(J64, B72,A72)` |
| K72 | `=CALC_ACCURACY_UPDATED(K64, B72,A72)` |
| L72 | `=CALC_ACCURACY_UPDATED(L64, B72, A72)` |
| M72 | `=CALC_ACCURACY_UPDATED(M64, B72, A72)` |
| N72 | `=CALC_ACCURACY_UPDATED(N64, B72, A72)` |
| O72 | `=CALC_ACCURACY_UPDATED(O64, B72, A72)` |
| P72 | `=CALC_ACCURACY_UPDATED(P64, B72, A72)` |
| Q72 | `=CALC_ACCURACY_UPDATED(Q64, B72, A72)` |
| C73 | `=CALC_ACCURACY_UPDATED(C64, B73,A73)` |
| D73 | `=CALC_ACCURACY_UPDATED(D64,B73,A73)` |
| E73 | `=CALC_ACCURACY_UPDATED(E64, B73, A73)` |
| F73 | `=CALC_ACCURACY_UPDATED(F64, B73,A73)` |
| G73 | `=CALC_ACCURACY_UPDATED(G64, B73,A73)` |
| H73 | `=CALC_ACCURACY_UPDATED(H64, B73,A73)` |
| I73 | `=CALC_ACCURACY_UPDATED(I64, B73,A73)` |
| J73 | `=CALC_ACCURACY_UPDATED(J64, B73,A73)` |
| K73 | `=CALC_ACCURACY_UPDATED(K64, B73,A73)` |
| L73 | `=CALC_ACCURACY_UPDATED(L64, B73, A73)` |
| M73 | `=CALC_ACCURACY_UPDATED(M64, B73, A73)` |
| N73 | `=CALC_ACCURACY_UPDATED(N64, B73, A73)` |
| O73 | `=CALC_ACCURACY_UPDATED(O64, B73, A73)` |
| P73 | `=CALC_ACCURACY_UPDATED(P64, B73, A73)` |
| Q73 | `=CALC_ACCURACY_UPDATED(Q64, B73, A73)` |
| C74 | `=CALC_ACCURACY_UPDATED(C64, B74,A74)` |
| D74 | `=CALC_ACCURACY_UPDATED(D64,B74,A74)` |
| E74 | `=CALC_ACCURACY_UPDATED(E64, B74, A74)` |
| F74 | `=CALC_ACCURACY_UPDATED(F64, B74,A74)` |
| G74 | `=CALC_ACCURACY_UPDATED(G64, B74,A74)` |
| H74 | `=CALC_ACCURACY_UPDATED(H64, B74,A74)` |
| I74 | `=CALC_ACCURACY_UPDATED(I64, B74,A74)` |
| J74 | `=CALC_ACCURACY_UPDATED(J64, B74,A74)` |
| K74 | `=CALC_ACCURACY_UPDATED(K64, B74,A74)` |
| L74 | `=CALC_ACCURACY_UPDATED(L64, B74, A74)` |
| M74 | `=CALC_ACCURACY_UPDATED(M64, B74, A74)` |
| N74 | `=CALC_ACCURACY_UPDATED(N64, B74, A74)` |
| O74 | `=CALC_ACCURACY_UPDATED(O64, B74, A74)` |
| P74 | `=CALC_ACCURACY_UPDATED(P64, B74, A74)` |
| Q74 | `=CALC_ACCURACY_UPDATED(Q64, B74, A74)` |
| C75 | `=CALC_ACCURACY_UPDATED(C64, B75,A75)` |
| D75 | `=CALC_ACCURACY_UPDATED(D64,B75,A75)` |
| E75 | `=CALC_ACCURACY_UPDATED(E64, B75, A75)` |
| F75 | `=CALC_ACCURACY_UPDATED(F64, B75,A75)` |
| G75 | `=CALC_ACCURACY_UPDATED(G64, B75,A75)` |
| H75 | `=CALC_ACCURACY_UPDATED(H64, B75,A75)` |
| I75 | `=CALC_ACCURACY_UPDATED(I64, B75,A75)` |
| J75 | `=CALC_ACCURACY_UPDATED(J64, B75,A75)` |
| K75 | `=CALC_ACCURACY_UPDATED(K64, B75,A75)` |
| L75 | `=CALC_ACCURACY_UPDATED(L64, B75, A75)` |
| M75 | `=CALC_ACCURACY_UPDATED(M64, B75, A75)` |
| N75 | `=CALC_ACCURACY_UPDATED(N64, B75, A75)` |
| O75 | `=CALC_ACCURACY_UPDATED(O64, B75, A75)` |
| P75 | `=CALC_ACCURACY_UPDATED(P64, B75, A75)` |
| Q75 | `=CALC_ACCURACY_UPDATED(Q64, B75, A75)` |
| C76 | `=CALC_ACCURACY_UPDATED(C64, B76,A76)` |
| D76 | `=CALC_ACCURACY_UPDATED(D64,B76,A76)` |
| E76 | `=CALC_ACCURACY_UPDATED(E64, B76, A76)` |
| F76 | `=CALC_ACCURACY_UPDATED(F64, B76,A76)` |
| G76 | `=CALC_ACCURACY_UPDATED(G64, B76,A76)` |
| H76 | `=CALC_ACCURACY_UPDATED(H64, B76,A76)` |
| I76 | `=CALC_ACCURACY_UPDATED(I64, B76,A76)` |
| J76 | `=CALC_ACCURACY_UPDATED(J64, B76,A76)` |
| K76 | `=CALC_ACCURACY_UPDATED(K64, B76,A76)` |
| L76 | `=CALC_ACCURACY_UPDATED(L64, B76, A76)` |
| M76 | `=CALC_ACCURACY_UPDATED(M64, B76, A76)` |
| N76 | `=CALC_ACCURACY_UPDATED(N64, B76, A76)` |
| O76 | `=CALC_ACCURACY_UPDATED(O64, B76, A76)` |
| P76 | `=CALC_ACCURACY_UPDATED(P64, B76, A76)` |
| Q76 | `=CALC_ACCURACY_UPDATED(Q64, B76, A76)` |
| C77 | `=CALC_ACCURACY_UPDATED(C64, B77,A77)` |
| D77 | `=CALC_ACCURACY_UPDATED(D64,B77,A77)` |
| E77 | `=CALC_ACCURACY_UPDATED(E64, B77, A77)` |
| F77 | `=CALC_ACCURACY_UPDATED(F64, B77,A77)` |
| G77 | `=CALC_ACCURACY_UPDATED(G64, B77,A77)` |
| H77 | `=CALC_ACCURACY_UPDATED(H64, B77,A77)` |
| I77 | `=CALC_ACCURACY_UPDATED(I64, B77,A77)` |
| J77 | `=CALC_ACCURACY_UPDATED(J64, B77,A77)` |
| K77 | `=CALC_ACCURACY_UPDATED(K64, B77,A77)` |
| L77 | `=CALC_ACCURACY_UPDATED(L64, B77, A77)` |
| M77 | `=CALC_ACCURACY_UPDATED(M64, B77, A77)` |
| N77 | `=CALC_ACCURACY_UPDATED(N64, B77, A77)` |
| O77 | `=CALC_ACCURACY_UPDATED(O64, B77, A77)` |
| P77 | `=CALC_ACCURACY_UPDATED(P64, B77, A77)` |
| Q77 | `=CALC_ACCURACY_UPDATED(Q64, B77, A77)` |
| C78 | `=CALC_ACCURACY_UPDATED(C64,B78,A78)` |
| D78 | `=CALC_ACCURACY_UPDATED(D64,B78,A78)` |
| E78 | `=CALC_ACCURACY_UPDATED(E64, B78, A78)` |
| F78 | `=CALC_ACCURACY_UPDATED(F64, B78,A78)` |
| G78 | `=CALC_ACCURACY_UPDATED(G64, B78,A78)` |
| H78 | `=CALC_ACCURACY_UPDATED(H64, B78,A78)` |
| I78 | `=CALC_ACCURACY_UPDATED(I64, B78,A78)` |
| J78 | `=CALC_ACCURACY_UPDATED(J64, B78,A78)` |
| K78 | `=CALC_ACCURACY_UPDATED(K64, B78,A78)` |
| L78 | `=CALC_ACCURACY_UPDATED(L64, B78, A78)` |
| M78 | `=CALC_ACCURACY_UPDATED(M64, B78, A78)` |
| N78 | `=CALC_ACCURACY_UPDATED(N64, B78, A78)` |
| O78 | `=CALC_ACCURACY_UPDATED(O64, B78, A78)` |
| P78 | `=CALC_ACCURACY_UPDATED(P64, B78, A78)` |
| Q78 | `=CALC_ACCURACY_UPDATED(Q64, B78, A78)` |
| C79 | `=CALC_ACCURACY_UPDATED(C64, "", "Diagnosis")` |
| D79 | `=CALC_ACCURACY(D64, "", "Diagnosis")` |
| E79 | `=CALC_ACCURACY_UPDATED(E64, B79, "Diagnosis")` |
| F79 | `=CALC_ACCURACY_UPDATED(F64, B79, "Diagnosis")` |
| G79 | `=CALC_ACCURACY_UPDATED(G64, B79, "Diagnosis")` |
| H79 | `=CALC_ACCURACY_UPDATED(H64, B79, "Diagnosis")` |
| I79 | `=CALC_ACCURACY_UPDATED(I64, B79, "Diagnosis")` |
| J79 | `=CALC_ACCURACY_UPDATED(J64, B79, "Diagnosis")` |
| K79 | `=CALC_ACCURACY_UPDATED(K64, B79, "Diagnosis")` |
| L79 | `=CALC_ACCURACY_UPDATED(L64, B79, "Diagnosis")` |
| M79 | `=CALC_ACCURACY_UPDATED(M64, B79, "Diagnosis")` |
| N79 | `=CALC_ACCURACY_UPDATED(N64, B79, "Diagnosis")` |
| O79 | `=CALC_ACCURACY_UPDATED(O64, B79, "Diagnosis")` |
| P79 | `=CALC_ACCURACY_UPDATED(P64, B79, "Diagnosis")` |
| Q79 | `=CALC_ACCURACY_UPDATED(Q64, B79, "Diagnosis")` |
| C80 | `=CALC_ACCURACY_UPDATED(C64, "", "Management")` |
| D80 | `=CALC_ACCURACY_UPDATED(D64, "", "Management")` |
| E80 | `=CALC_ACCURACY_UPDATED(E64, B80, "Management")` |
| F80 | `=CALC_ACCURACY_UPDATED(F64, B80, "Management")` |
| G80 | `=CALC_ACCURACY_UPDATED(G64, B80, "Management")` |
| H80 | `=CALC_ACCURACY_UPDATED(H64, B80, "Management")` |
| I80 | `=CALC_ACCURACY_UPDATED(I64, B80, "Management")` |
| J80 | `=CALC_ACCURACY_UPDATED(J64, B80, "Management")` |
| K80 | `=CALC_ACCURACY_UPDATED(K64, B80, "Management")` |
| L80 | `=CALC_ACCURACY_UPDATED(L64, B80, "Management")` |
| M80 | `=CALC_ACCURACY_UPDATED(M64, B80, "Management")` |
| N80 | `=CALC_ACCURACY_UPDATED(N64, B80, "Management")` |
| O80 | `=CALC_ACCURACY_UPDATED(O64, B80, "Management")` |
| P80 | `=CALC_ACCURACY_UPDATED(P64, B80, "Management")` |
| Q80 | `=CALC_ACCURACY_UPDATED(Q64, B80, "Management")` |
| C81 | `=COMBINE_DIAG_MANAG(C79, C80, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| D81 | `=COMBINE_DIAG_MANAG(D79, D80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| E81 | `=COMBINE_DIAG_MANAG(E79, E80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| F81 | `=COMBINE_DIAG_MANAG(F79, F80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| G81 | `=COMBINE_DIAG_MANAG(G79, G80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| H81 | `=COMBINE_DIAG_MANAG(H79, H80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| I81 | `=COMBINE_DIAG_MANAG(I79, I80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| J81 | `=COMBINE_DIAG_MANAG(J79, J80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| K81 | `=COMBINE_DIAG_MANAG(K79, K80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| L81 | `=COMBINE_DIAG_MANAG(L79, L80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| M81 | `=COMBINE_DIAG_MANAG(M79, M80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| N81 | `=COMBINE_DIAG_MANAG(N79, N80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| O81 | `=COMBINE_DIAG_MANAG(O79, O80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| P81 | `=COMBINE_DIAG_MANAG(P79, P80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| Q81 | `=COMBINE_DIAG_MANAG(Q79, Q80, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| C82 | `=CALC_ACCURACY_UPDATED(C64, "", "")` |
| D82 | `=CALC_ACCURACY_UPDATED(D64, "", "")` |
| E82 | `=CALC_ACCURACY_UPDATED(E64, "", "")` |
| F82 | `=CALC_ACCURACY_UPDATED(F64, "", "")` |
| G82 | `=CALC_ACCURACY_UPDATED(G64, "", "")` |
| H82 | `=CALC_ACCURACY_UPDATED(H64, "", "")` |
| I82 | `=CALC_ACCURACY_UPDATED(I64, "", "")` |
| J82 | `=CALC_ACCURACY_UPDATED(J64, "", "")` |
| K82 | `=CALC_ACCURACY_UPDATED(K64, "", "")` |
| L82 | `=CALC_ACCURACY_UPDATED(L64, "", "")` |
| M82 | `=CALC_ACCURACY_UPDATED(M64, "", "")` |
| N82 | `=CALC_ACCURACY_UPDATED(N64, "", "")` |
| O82 | `=CALC_ACCURACY_UPDATED(O64, "", "")` |
| P82 | `=CALC_ACCURACY_UPDATED(P64, "", "")` |
| Q82 | `=CALC_ACCURACY_UPDATED(Q64, "", "")` |

### OmarOpthalmologyUpdated (769 formulas)

| Cell | Formula |
|------|--------|
| C16 | `=CALC_ACCURACY_UPDATED(C13, B16, A16)` |
| D16 | `=CALC_ACCURACY_UPDATED(D13, B16,A16)` |
| E16 | `=CALC_ACCURACY_UPDATED(E13, B16, A16)` |
| F16 | `=CALC_ACCURACY_UPDATED(F13, B16, A16)` |
| G16 | `=CALC_ACCURACY_UPDATED(G13, #REF!, #REF!)` |
| H16 | `=CALC_ACCURACY_UPDATED(H13, B16, A16)` |
| I16 | `=CALC_ACCURACY_UPDATED(I13,B16,A16)` |
| J16 | `=CALC_ACCURACY_UPDATED(J13, B16, A16)` |
| K16 | `=CALC_ACCURACY_UPDATED(K13, B16, A16)` |
| M16 | `=CALC_ACCURACY_UPDATED(M13, #REF!,D16)` |
| N16 | `=CALC_ACCURACY_UPDATED(N13, B16,A16)` |
| O16 | `=CALC_ACCURACY_UPDATED(O13, B16,A16)` |
| P16 | `=MAX(M16:P16)` |
| Q16 | `=CALC_ACCURACY_UPDATED(Q13, B16,A16)` |
| R16 | `=CALC_ACCURACY_UPDATED(R13, B16,A16)` |
| S16 | `=CALC_ACCURACY_UPDATED(S13, B16,A16)` |
| T16 | `=CALC_ACCURACY_UPDATED(T13, B16, A16)` |
| C17 | `=CALC_ACCURACY_UPDATED(C13, B17, A17)` |
| D17 | `=CALC_ACCURACY_UPDATED(D13, B17,A17)` |
| E17 | `=CALC_ACCURACY_UPDATED(E13, B17, A17)` |
| F17 | `=CALC_ACCURACY_UPDATED(F13, B17, A17)` |
| G17 | `=CALC_ACCURACY_UPDATED(G13, #REF!, #REF!)` |
| H17 | `=CALC_ACCURACY_UPDATED(H13, B17, A17)` |
| I17 | `=CALC_ACCURACY_UPDATED(I13,B17,A17)` |
| J17 | `=CALC_ACCURACY_UPDATED(J13, B17, A17)` |
| K17 | `=CALC_ACCURACY_UPDATED(K13, B17, A17)` |
| M17 | `=CALC_ACCURACY_UPDATED(M13, #REF!,D17)` |
| N17 | `=CALC_ACCURACY_UPDATED(N13, B17,A17)` |
| O17 | `=CALC_ACCURACY_UPDATED(O13, B17,A17)` |
| P17 | `=MAX(M17:P17)` |
| Q17 | `=CALC_ACCURACY_UPDATED(Q13, B17,A17)` |
| R17 | `=CALC_ACCURACY_UPDATED(R13, B17,A17)` |
| S17 | `=CALC_ACCURACY_UPDATED(S13, B17,A17)` |
| T17 | `=CALC_ACCURACY_UPDATED(T13, B17, A17)` |
| C18 | `=CALC_ACCURACY_UPDATED(C13, B18, A18)` |
| D18 | `=CALC_ACCURACY_UPDATED(D13, B18,A18)` |
| E18 | `=CALC_ACCURACY_UPDATED(E13, B18, A18)` |
| F18 | `=CALC_ACCURACY_UPDATED(F13, B18, A18)` |
| G18 | `=CALC_ACCURACY_UPDATED(G13, #REF!, #REF!)` |
| H18 | `=CALC_ACCURACY_UPDATED(H13, B18, A18)` |
| I18 | `=CALC_ACCURACY_UPDATED(I13,B18,A18)` |
| J18 | `=CALC_ACCURACY_UPDATED(J13, B18, A18)` |
| K18 | `=CALC_ACCURACY_UPDATED(K13, B18, A18)` |
| M18 | `=CALC_ACCURACY_UPDATED(M13, #REF!,D18)` |
| N18 | `=CALC_ACCURACY_UPDATED(N13, B18,A18)` |
| O18 | `=CALC_ACCURACY_UPDATED(O13, B18,A18)` |
| P18 | `=MAX(M18:P18)` |
| Q18 | `=CALC_ACCURACY_UPDATED(Q13, B18,A18)` |
| R18 | `=CALC_ACCURACY_UPDATED(R13, B18,A18)` |
| S18 | `=CALC_ACCURACY_UPDATED(S13, B18,A18)` |
| T18 | `=CALC_ACCURACY_UPDATED(T13, B18, A18)` |
| C19 | `=CALC_ACCURACY_UPDATED(C13, B19, A19)` |
| D19 | `=CALC_ACCURACY_UPDATED(D13, B19,A19)` |
| E19 | `=CALC_ACCURACY_UPDATED(E13, B19, A19)` |
| F19 | `=CALC_ACCURACY_UPDATED(F13, B19, A19)` |
| G19 | `=CALC_ACCURACY_UPDATED(G13, #REF!, #REF!)` |
| H19 | `=CALC_ACCURACY_UPDATED(H13, B19, A19)` |
| I19 | `=CALC_ACCURACY_UPDATED(I13,B19,A19)` |
| J19 | `=CALC_ACCURACY_UPDATED(J13, B19, A19)` |
| K19 | `=CALC_ACCURACY_UPDATED(K13, B19, A19)` |
| M19 | `=CALC_ACCURACY_UPDATED(M13, #REF!,D19)` |
| N19 | `=CALC_ACCURACY_UPDATED(N13, B19,A19)` |
| O19 | `=CALC_ACCURACY_UPDATED(O13, B19,A19)` |
| P19 | `=MAX(M19:P19)` |
| Q19 | `=CALC_ACCURACY_UPDATED(Q13, B19,A19)` |
| R19 | `=CALC_ACCURACY_UPDATED(R13, B19,A19)` |
| S19 | `=CALC_ACCURACY_UPDATED(S13, B19,A19)` |
| T19 | `=CALC_ACCURACY_UPDATED(T13, B19, A19)` |
| C20 | `=CALC_ACCURACY_UPDATED(C13, B20, A20)` |
| D20 | `=CALC_ACCURACY_UPDATED(D13, B20,A20)` |
| E20 | `=CALC_ACCURACY_UPDATED(E13, B20, A20)` |
| F20 | `=CALC_ACCURACY_UPDATED(F13, B20, A20)` |
| G20 | `=CALC_ACCURACY_UPDATED(G13, #REF!, #REF!)` |
| H20 | `=CALC_ACCURACY_UPDATED(H13, B20, A20)` |
| I20 | `=CALC_ACCURACY_UPDATED(I13,B20,A20)` |
| J20 | `=CALC_ACCURACY_UPDATED(J13, B20, A20)` |
| K20 | `=CALC_ACCURACY_UPDATED(K13, B20, A20)` |
| M20 | `=CALC_ACCURACY_UPDATED(M13, #REF!,D20)` |
| N20 | `=CALC_ACCURACY_UPDATED(N13, B20,A20)` |
| O20 | `=CALC_ACCURACY_UPDATED(O13, B20,A20)` |
| P20 | `=MAX(M20:P20)` |
| Q20 | `=CALC_ACCURACY_UPDATED(Q13, B20,A20)` |
| R20 | `=CALC_ACCURACY_UPDATED(R13, B20,A20)` |
| S20 | `=CALC_ACCURACY_UPDATED(S13, B20,A20)` |
| T20 | `=CALC_ACCURACY_UPDATED(T13, B20, A20)` |
| C21 | `=CALC_ACCURACY_UPDATED(C13, B21, A21)` |
| D21 | `=CALC_ACCURACY_UPDATED(D13, B21,A21)` |
| E21 | `=CALC_ACCURACY_UPDATED(E13, B21, A21)` |
| F21 | `=CALC_ACCURACY_UPDATED(F13, B21, A21)` |
| G21 | `=CALC_ACCURACY_UPDATED(G13, #REF!, #REF!)` |
| H21 | `=CALC_ACCURACY_UPDATED(H13, B21, A21)` |
| I21 | `=CALC_ACCURACY_UPDATED(I13,B21,A21)` |
| J21 | `=CALC_ACCURACY_UPDATED(J13, B21, A21)` |
| K21 | `=CALC_ACCURACY_UPDATED(K13, B21, A21)` |
| M21 | `=CALC_ACCURACY_UPDATED(M13, #REF!,D21)` |
| N21 | `=CALC_ACCURACY_UPDATED(N13, B21,A21)` |
| O21 | `=CALC_ACCURACY_UPDATED(O13, B21,A21)` |
| P21 | `=MAX(M21:P21)` |
| Q21 | `=CALC_ACCURACY_UPDATED(Q13, B21,A21)` |
| R21 | `=CALC_ACCURACY_UPDATED(R13, B21,A21)` |
| S21 | `=CALC_ACCURACY_UPDATED(S13, B21,A21)` |
| T21 | `=CALC_ACCURACY_UPDATED(T13, B21, A21)` |
| C22 | `=CALC_ACCURACY_UPDATED(C13, B22, A22)` |
| D22 | `=CALC_ACCURACY_UPDATED(D13, B22,A22)` |
| E22 | `=CALC_ACCURACY_UPDATED(E13, B22, A22)` |
| F22 | `=CALC_ACCURACY_UPDATED(F13, B22, A22)` |
| G22 | `=CALC_ACCURACY_UPDATED(G13, #REF!, #REF!)` |
| H22 | `=CALC_ACCURACY_UPDATED(H13, B22, A22)` |
| I22 | `=CALC_ACCURACY_UPDATED(I13,B22,A22)` |
| J22 | `=CALC_ACCURACY_UPDATED(J13, B22, A22)` |
| K22 | `=CALC_ACCURACY_UPDATED(K13, B22, A22)` |
| M22 | `=CALC_ACCURACY_UPDATED(M13, #REF!,D22)` |
| N22 | `=CALC_ACCURACY_UPDATED(N13, B22,A22)` |
| O22 | `=CALC_ACCURACY_UPDATED(O13, B22,A22)` |
| P22 | `=MAX(M22:P22)` |
| Q22 | `=CALC_ACCURACY_UPDATED(Q13, B22,A22)` |
| R22 | `=CALC_ACCURACY_UPDATED(R13, B22,A22)` |
| S22 | `=CALC_ACCURACY_UPDATED(S13, B22,A22)` |
| T22 | `=CALC_ACCURACY_UPDATED(T13, B22, A22)` |
| C23 | `=CALC_ACCURACY_UPDATED(C13, B23, A23)` |
| D23 | `=CALC_ACCURACY_UPDATED(D13, B23,A23)` |
| E23 | `=CALC_ACCURACY_UPDATED(E13, B23, A23)` |
| F23 | `=CALC_ACCURACY_UPDATED(F13, B23, A23)` |
| G23 | `=CALC_ACCURACY_UPDATED(G13, #REF!, #REF!)` |
| H23 | `=CALC_ACCURACY_UPDATED(H13, B23, A23)` |
| I23 | `=CALC_ACCURACY_UPDATED(I13,B23,A23)` |
| J23 | `=CALC_ACCURACY_UPDATED(J13, B23, A23)` |
| K23 | `=CALC_ACCURACY_UPDATED(K13, B23, A23)` |
| M23 | `=CALC_ACCURACY_UPDATED(M13, #REF!,D23)` |
| N23 | `=CALC_ACCURACY_UPDATED(N13, B23,A23)` |
| O23 | `=CALC_ACCURACY_UPDATED(O13, B23,A23)` |
| P23 | `=MAX(M23:P23)` |
| Q23 | `=CALC_ACCURACY_UPDATED(Q13, B23,A23)` |
| R23 | `=CALC_ACCURACY_UPDATED(R13, B23,A23)` |
| S23 | `=CALC_ACCURACY_UPDATED(S13, B23,A23)` |
| T23 | `=CALC_ACCURACY_UPDATED(T13, B23, A23)` |
| C24 | `=CALC_ACCURACY_UPDATED(C13, B24, A24)` |
| D24 | `=CALC_ACCURACY_UPDATED(D13, B24,A24)` |
| E24 | `=CALC_ACCURACY_UPDATED(E13, B24, A24)` |
| F24 | `=CALC_ACCURACY_UPDATED(F13, B24, A24)` |
| G24 | `=CALC_ACCURACY_UPDATED(G13, #REF!, #REF!)` |
| H24 | `=CALC_ACCURACY_UPDATED(H13, B24, A24)` |
| I24 | `=CALC_ACCURACY_UPDATED(I13,B24,A24)` |
| J24 | `=CALC_ACCURACY_UPDATED(J13, B24, A24)` |
| K24 | `=CALC_ACCURACY_UPDATED(K13, B24, A24)` |
| M24 | `=CALC_ACCURACY_UPDATED(M13, #REF!,D24)` |
| N24 | `=CALC_ACCURACY_UPDATED(N13, B24,A24)` |
| O24 | `=CALC_ACCURACY_UPDATED(O13, B24,A24)` |
| P24 | `=MAX(M24:P24)` |
| Q24 | `=CALC_ACCURACY_UPDATED(Q13, B24,A24)` |
| R24 | `=CALC_ACCURACY_UPDATED(R13, B24,A24)` |
| S24 | `=CALC_ACCURACY_UPDATED(S13, B24,A24)` |
| T24 | `=CALC_ACCURACY_UPDATED(T13, B24, A24)` |
| C25 | `=CALC_ACCURACY_UPDATED(C13, B25, A25)` |
| D25 | `=CALC_ACCURACY_UPDATED(D13, B25,A25)` |
| E25 | `=CALC_ACCURACY_UPDATED(E13, B25, A25)` |
| F25 | `=CALC_ACCURACY_UPDATED(F13, B25, A25)` |
| G25 | `=CALC_ACCURACY_UPDATED(G13, #REF!, #REF!)` |
| H25 | `=CALC_ACCURACY_UPDATED(H13, B25, A25)` |
| I25 | `=CALC_ACCURACY_UPDATED(I13,B25,A25)` |
| J25 | `=CALC_ACCURACY_UPDATED(J13, B25, A25)` |
| K25 | `=CALC_ACCURACY_UPDATED(K13, B25, A25)` |
| M25 | `=CALC_ACCURACY_UPDATED(M13, #REF!,D25)` |
| N25 | `=CALC_ACCURACY_UPDATED(N13, B25,A25)` |
| O25 | `=CALC_ACCURACY_UPDATED(O13, B25,A25)` |
| P25 | `=MAX(M25:P25)` |
| Q25 | `=CALC_ACCURACY_UPDATED(Q13, B25,A25)` |
| R25 | `=CALC_ACCURACY_UPDATED(R13, B25,A25)` |
| S25 | `=CALC_ACCURACY_UPDATED(S13, B25,A25)` |
| T25 | `=CALC_ACCURACY_UPDATED(T13, B25, A25)` |
| C26 | `=CALC_ACCURACY_UPDATED(C13, B26, A26)` |
| D26 | `=CALC_ACCURACY_UPDATED(D13, B26,A26)` |
| E26 | `=CALC_ACCURACY_UPDATED(E13, B26, A26)` |
| F26 | `=CALC_ACCURACY_UPDATED(F13, B26, A26)` |
| G26 | `=CALC_ACCURACY_UPDATED(G13, #REF!, #REF!)` |
| H26 | `=CALC_ACCURACY_UPDATED(H13, B26, A26)` |
| I26 | `=CALC_ACCURACY_UPDATED(I13,B26,A26)` |
| J26 | `=CALC_ACCURACY_UPDATED(J13, B26, A26)` |
| K26 | `=CALC_ACCURACY_UPDATED(K13, B26, A26)` |
| M26 | `=CALC_ACCURACY_UPDATED(M13, #REF!,D26)` |
| N26 | `=CALC_ACCURACY_UPDATED(N13, B26,A26)` |
| O26 | `=CALC_ACCURACY_UPDATED(O13, B26,A26)` |
| P26 | `=MAX(M26:P26)` |
| Q26 | `=CALC_ACCURACY_UPDATED(Q13, B26,A26)` |
| R26 | `=CALC_ACCURACY_UPDATED(R13, B26,A26)` |
| S26 | `=CALC_ACCURACY_UPDATED(S13, B26,A26)` |
| T26 | `=CALC_ACCURACY_UPDATED(T13, B26, A26)` |
| C27 | `=CALC_ACCURACY_UPDATED(C13, B27, A27)` |
| D27 | `=CALC_ACCURACY_UPDATED(D13,B27,A27)` |
| E27 | `=CALC_ACCURACY_UPDATED(E13, B27, A27)` |
| F27 | `=CALC_ACCURACY_UPDATED(F13, B27, A27)` |
| G27 | `=CALC_ACCURACY_UPDATED(G13, #REF!, #REF!)` |
| H27 | `=CALC_ACCURACY_UPDATED(H13, B27, A27)` |
| I27 | `=CALC_ACCURACY_UPDATED(I13,B27,A27)` |
| J27 | `=CALC_ACCURACY_UPDATED(J13, B27, A27)` |
| K27 | `=CALC_ACCURACY_UPDATED(K13, B27, A27)` |
| M27 | `=CALC_ACCURACY_UPDATED(M13, #REF!,D27)` |
| N27 | `=CALC_ACCURACY_UPDATED(N13, B27,A27)` |
| O27 | `=CALC_ACCURACY_UPDATED(O13, B27,A27)` |
| P27 | `=MAX(M27:P27)` |
| Q27 | `=CALC_ACCURACY_UPDATED(Q13, B27,A27)` |
| R27 | `=CALC_ACCURACY_UPDATED(R13, B27,A27)` |
| S27 | `=CALC_ACCURACY_UPDATED(S13, B27,A27)` |
| T27 | `=CALC_ACCURACY_UPDATED(T13, B27, A27)` |
| C28 | `=CALC_ACCURACY_UPDATED(C13, B28, "Diagnosis")` |
| D28 | `=CALC_ACCURACY_UPDATED(D13, "", "Diagnosis")` |
| E28 | `=CALC_ACCURACY_UPDATED(E13, B28, "Diagnosis")` |
| F28 | `=CALC_ACCURACY_UPDATED(F13, B28, "Diagnosis")` |
| G28 | `=CALC_ACCURACY_UPDATED(G13, #REF!, "Diagnosis")` |
| H28 | `=CALC_ACCURACY_UPDATED(H13, B28, "Diagnosis")` |
| I28 | `=CALC_ACCURACY(I13, "", "Diagnosis")` |
| J28 | `=CALC_ACCURACY_UPDATED(J13, B28, "Diagnosis")` |
| K28 | `=CALC_ACCURACY_UPDATED(K13, B28, "Diagnosis")` |
| M28 | `=CALC_ACCURACY_UPDATED(M13, #REF!, "Diagnosis")` |
| N28 | `=CALC_ACCURACY_UPDATED(N13, B28, "Diagnosis")` |
| O28 | `=CALC_ACCURACY_UPDATED(O13, B28, "Diagnosis")` |
| P28 | `=MAX(M28:P28)` |
| Q28 | `=CALC_ACCURACY_UPDATED(Q13, B28, "Diagnosis")` |
| R28 | `=CALC_ACCURACY_UPDATED(R13, B28, "Diagnosis")` |
| S28 | `=CALC_ACCURACY_UPDATED(S13, B28, "Diagnosis")` |
| T28 | `=CALC_ACCURACY_UPDATED(T13, B28, "Diagnosis")` |
| C29 | `=CALC_ACCURACY_UPDATED(C13, B29, "Management")` |
| D29 | `=CALC_ACCURACY_UPDATED(D13, "", "Management")` |
| E29 | `=CALC_ACCURACY_UPDATED(E13, B29, "Management")` |
| F29 | `=CALC_ACCURACY_UPDATED(F13, B29, "Management")` |
| G29 | `=CALC_ACCURACY_UPDATED(G13, #REF!, "Management")` |
| H29 | `=CALC_ACCURACY_UPDATED(H13, B29, "Management")` |
| I29 | `=CALC_ACCURACY_UPDATED(I13, "", "Management")` |
| J29 | `=CALC_ACCURACY_UPDATED(J13, B29, "Management")` |
| K29 | `=CALC_ACCURACY_UPDATED(K13, B29, "Management")` |
| M29 | `=CALC_ACCURACY_UPDATED(M13, #REF!, "Management")` |
| N29 | `=CALC_ACCURACY_UPDATED(N13, B29, "Management")` |
| O29 | `=CALC_ACCURACY_UPDATED(O13, B29, "Management")` |
| P29 | `=MAX(M29:P29)` |
| Q29 | `=CALC_ACCURACY_UPDATED(Q13, B29, "Management")` |
| R29 | `=CALC_ACCURACY_UPDATED(R13, B29, "Management")` |
| S29 | `=CALC_ACCURACY_UPDATED(S13, B29, "Management")` |
| T29 | `=CALC_ACCURACY_UPDATED(T13, B29, "Management")` |
| C30 | `=COMBINE_DIAG_MANAG(C28, C29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| D30 | `=COMBINE_DIAG_MANAG(D28, D29, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| E30 | `=COMBINE_DIAG_MANAG(E28, E29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| F30 | `=COMBINE_DIAG_MANAG(F28, F29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| G30 | `=COMBINE_DIAG_MANAG(G28, G29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| H30 | `=COMBINE_DIAG_MANAG(H28, H29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| I30 | `=COMBINE_DIAG_MANAG(I28, I29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| J30 | `=COMBINE_DIAG_MANAG(J28, J29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| K30 | `=COMBINE_DIAG_MANAG(K28, K29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| M30 | `=COMBINE_DIAG_MANAG(M28, M29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| N30 | `=COMBINE_DIAG_MANAG(N28, N29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| O30 | `=COMBINE_DIAG_MANAG(O28, O29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| P30 | `=MAX(M30:P30)` |
| Q30 | `=COMBINE_DIAG_MANAG(Q28, Q29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| R30 | `=COMBINE_DIAG_MANAG(R28, R29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| S30 | `=COMBINE_DIAG_MANAG(S28, S29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| T30 | `=COMBINE_DIAG_MANAG(T28, T29, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| C31 | `=CALC_ACCURACY_UPDATED(C13, "", "")` |
| D31 | `=CALC_ACCURACY_UPDATED(D13, "", "")` |
| E31 | `=CALC_ACCURACY_UPDATED(E13, "", "")` |
| F31 | `=CALC_ACCURACY_UPDATED(F13, "", "")` |
| G31 | `=CALC_ACCURACY_UPDATED(G13, "", "")` |
| H31 | `=CALC_ACCURACY_UPDATED(H13, "", "")` |
| I31 | `=CALC_ACCURACY_UPDATED(I13, "", "")` |
| J31 | `=CALC_ACCURACY_UPDATED(J13, "", "")` |
| K31 | `=CALC_ACCURACY_UPDATED(K13, "", "")` |
| M31 | `=CALC_ACCURACY_UPDATED(M13, "", "")` |
| N31 | `=CALC_ACCURACY_UPDATED(N13, "", "")` |
| O31 | `=CALC_ACCURACY_UPDATED(O13, "", "")` |
| P31 | `=MAX(M31:P31)` |
| Q31 | `=CALC_ACCURACY_UPDATED(Q13, "", "")` |
| R31 | `=CALC_ACCURACY_UPDATED(R13, "", "")` |
| S31 | `=CALC_ACCURACY_UPDATED(S13, "", "")` |
| T31 | `=CALC_ACCURACY_UPDATED(T13, "", "")` |
| C40 | `=CALC_ACCURACY_UPDATED(C37, B40, A40)` |
| D40 | `=CALC_ACCURACY_UPDATED(D37, B40,A40)` |
| E40 | `=CALC_ACCURACY_UPDATED(E37, B40, A40)` |
| F40 | `=CALC_ACCURACY_UPDATED(F37,B40,A40)` |
| G40 | `=CALC_ACCURACY_UPDATED(G37, B40, A40)` |
| H40 | `=CALC_ACCURACY_UPDATED(H37, B40, A40)` |
| I40 | `=CALC_ACCURACY_UPDATED(I37, B40,A40)` |
| J40 | `=CALC_ACCURACY_UPDATED(J37, B40,A40)` |
| K40 | `=CALC_ACCURACY_UPDATED(K37, B40,A40)` |
| M40 | `=CALC_ACCURACY_UPDATED(M37, B40,A40)` |
| N40 | `=CALC_ACCURACY_UPDATED(N37, B40,A40)` |
| O40 | `=CALC_ACCURACY_UPDATED(O37, B40,A40)` |
| Q40 | `=CALC_ACCURACY_UPDATED(Q37, B40, A40)` |
| R40 | `=CALC_ACCURACY_UPDATED(R37, B40, A40)` |
| S40 | `=CALC_ACCURACY_UPDATED(S37, B40, A40)` |
| C41 | `=CALC_ACCURACY_UPDATED(C37, B41, A41)` |
| D41 | `=CALC_ACCURACY_UPDATED(D37, B41,A41)` |
| E41 | `=CALC_ACCURACY_UPDATED(E37, B41, A41)` |
| F41 | `=CALC_ACCURACY_UPDATED(F37,B41,A41)` |
| G41 | `=CALC_ACCURACY_UPDATED(G37, B41, A41)` |
| H41 | `=CALC_ACCURACY_UPDATED(H37, B41, A41)` |
| I41 | `=CALC_ACCURACY_UPDATED(I37, B41,A41)` |
| J41 | `=CALC_ACCURACY_UPDATED(J37, B41,A41)` |
| K41 | `=CALC_ACCURACY_UPDATED(K37, B41,A41)` |
| M41 | `=CALC_ACCURACY_UPDATED(M37, B41,A41)` |
| N41 | `=CALC_ACCURACY_UPDATED(N37, B41,A41)` |
| O41 | `=CALC_ACCURACY_UPDATED(O37, B41,A41)` |
| Q41 | `=CALC_ACCURACY_UPDATED(Q37, B41, A41)` |
| R41 | `=CALC_ACCURACY_UPDATED(R37, B41, A41)` |
| S41 | `=CALC_ACCURACY_UPDATED(S37, B41, A41)` |
| C42 | `=CALC_ACCURACY_UPDATED(C37, B42, A42)` |
| D42 | `=CALC_ACCURACY_UPDATED(D37, B42,A42)` |
| E42 | `=CALC_ACCURACY_UPDATED(E37, B42, A42)` |
| F42 | `=CALC_ACCURACY_UPDATED(F37,B42,A42)` |
| G42 | `=CALC_ACCURACY_UPDATED(G37, B42, A42)` |
| H42 | `=CALC_ACCURACY_UPDATED(H37, B42, A42)` |
| I42 | `=CALC_ACCURACY_UPDATED(I37, B42,A42)` |
| J42 | `=CALC_ACCURACY_UPDATED(J37, B42,A42)` |
| K42 | `=CALC_ACCURACY_UPDATED(K37, B42,A42)` |
| M42 | `=CALC_ACCURACY_UPDATED(M37, B42,A42)` |
| N42 | `=CALC_ACCURACY_UPDATED(N37, B42,A42)` |
| O42 | `=CALC_ACCURACY_UPDATED(O37, B42,A42)` |
| Q42 | `=CALC_ACCURACY_UPDATED(Q37, B42, A42)` |
| R42 | `=CALC_ACCURACY_UPDATED(R37, B42, A42)` |
| S42 | `=CALC_ACCURACY_UPDATED(S37, B42, A42)` |
| C43 | `=CALC_ACCURACY_UPDATED(C37, B43, A43)` |
| D43 | `=CALC_ACCURACY_UPDATED(D37, B43,A43)` |
| E43 | `=CALC_ACCURACY_UPDATED(E37, B43, A43)` |
| F43 | `=CALC_ACCURACY_UPDATED(F37,B43,A43)` |
| G43 | `=CALC_ACCURACY_UPDATED(G37, B43, A43)` |
| H43 | `=CALC_ACCURACY_UPDATED(H37, B43, A43)` |
| I43 | `=CALC_ACCURACY_UPDATED(I37, B43,A43)` |
| J43 | `=CALC_ACCURACY_UPDATED(J37, B43,A43)` |
| K43 | `=CALC_ACCURACY_UPDATED(K37, B43,A43)` |
| M43 | `=CALC_ACCURACY_UPDATED(M37, B43,A43)` |
| N43 | `=CALC_ACCURACY_UPDATED(N37, B43,A43)` |
| O43 | `=CALC_ACCURACY_UPDATED(O37, B43,A43)` |
| Q43 | `=CALC_ACCURACY_UPDATED(Q37, B43, A43)` |
| R43 | `=CALC_ACCURACY_UPDATED(R37, B43, A43)` |
| S43 | `=CALC_ACCURACY_UPDATED(S37, B43, A43)` |
| C44 | `=CALC_ACCURACY_UPDATED(C37, B44, A44)` |
| D44 | `=CALC_ACCURACY_UPDATED(D37, B44,A44)` |
| E44 | `=CALC_ACCURACY_UPDATED(E37, B44, A44)` |
| F44 | `=CALC_ACCURACY_UPDATED(F37,B44,A44)` |
| G44 | `=CALC_ACCURACY_UPDATED(G37, B44, A44)` |
| H44 | `=CALC_ACCURACY_UPDATED(H37, B44, A44)` |
| I44 | `=CALC_ACCURACY_UPDATED(I37, B44,A44)` |
| J44 | `=CALC_ACCURACY_UPDATED(J37, B44,A44)` |
| K44 | `=CALC_ACCURACY_UPDATED(K37, B44,A44)` |
| M44 | `=CALC_ACCURACY_UPDATED(M37, B44,A44)` |
| N44 | `=CALC_ACCURACY_UPDATED(N37, B44,A44)` |
| O44 | `=CALC_ACCURACY_UPDATED(O37, B44,A44)` |
| Q44 | `=CALC_ACCURACY_UPDATED(Q37, B44, A44)` |
| R44 | `=CALC_ACCURACY_UPDATED(R37, B44, A44)` |
| S44 | `=CALC_ACCURACY_UPDATED(S37, B44, A44)` |
| C45 | `=CALC_ACCURACY_UPDATED(C37, B45, A45)` |
| D45 | `=CALC_ACCURACY_UPDATED(D37, B45,A45)` |
| E45 | `=CALC_ACCURACY_UPDATED(E37, B45, A45)` |
| F45 | `=CALC_ACCURACY_UPDATED(F37,B45,A45)` |
| G45 | `=CALC_ACCURACY_UPDATED(G37, B45, A45)` |
| H45 | `=CALC_ACCURACY_UPDATED(H37, B45, A45)` |
| I45 | `=CALC_ACCURACY_UPDATED(I37, B45,A45)` |
| J45 | `=CALC_ACCURACY_UPDATED(J37, B45,A45)` |
| K45 | `=CALC_ACCURACY_UPDATED(K37, B45,A45)` |
| M45 | `=CALC_ACCURACY_UPDATED(M37, B45,A45)` |
| N45 | `=CALC_ACCURACY_UPDATED(N37, B45,A45)` |
| O45 | `=CALC_ACCURACY_UPDATED(O37, B45,A45)` |
| Q45 | `=CALC_ACCURACY_UPDATED(Q37, B45, A45)` |
| R45 | `=CALC_ACCURACY_UPDATED(R37, B45, A45)` |
| S45 | `=CALC_ACCURACY_UPDATED(S37, B45, A45)` |
| C46 | `=CALC_ACCURACY_UPDATED(C37, B46, A46)` |
| D46 | `=CALC_ACCURACY_UPDATED(D37, B46,A46)` |
| E46 | `=CALC_ACCURACY_UPDATED(E37, B46, A46)` |
| F46 | `=CALC_ACCURACY_UPDATED(F37,B46,A46)` |
| G46 | `=CALC_ACCURACY_UPDATED(G37, B46, A46)` |
| H46 | `=CALC_ACCURACY_UPDATED(H37, B46, A46)` |
| I46 | `=CALC_ACCURACY_UPDATED(I37, B46,A46)` |
| J46 | `=CALC_ACCURACY_UPDATED(J37, B46,A46)` |
| K46 | `=CALC_ACCURACY_UPDATED(K37, B46,A46)` |
| M46 | `=CALC_ACCURACY_UPDATED(M37, B46,A46)` |
| N46 | `=CALC_ACCURACY_UPDATED(N37, B46,A46)` |
| O46 | `=CALC_ACCURACY_UPDATED(O37, B46,A46)` |
| Q46 | `=CALC_ACCURACY_UPDATED(Q37, B46, A46)` |
| R46 | `=CALC_ACCURACY_UPDATED(R37, B46, A46)` |
| S46 | `=CALC_ACCURACY_UPDATED(S37, B46, A46)` |
| C47 | `=CALC_ACCURACY_UPDATED(C37, B47, A47)` |
| D47 | `=CALC_ACCURACY_UPDATED(D37, B47,A47)` |
| E47 | `=CALC_ACCURACY_UPDATED(E37, B47, A47)` |
| F47 | `=CALC_ACCURACY_UPDATED(F37,B47,A47)` |
| G47 | `=CALC_ACCURACY_UPDATED(G37, B47, A47)` |
| H47 | `=CALC_ACCURACY_UPDATED(H37, B47, A47)` |
| I47 | `=CALC_ACCURACY_UPDATED(I37, B47,A47)` |
| J47 | `=CALC_ACCURACY_UPDATED(J37, B47,A47)` |
| K47 | `=CALC_ACCURACY_UPDATED(K37, B47,A47)` |
| M47 | `=CALC_ACCURACY_UPDATED(M37, B47,A47)` |
| N47 | `=CALC_ACCURACY_UPDATED(N37, B47,A47)` |
| O47 | `=CALC_ACCURACY_UPDATED(O37, B47,A47)` |
| Q47 | `=CALC_ACCURACY_UPDATED(Q37, B47, A47)` |
| R47 | `=CALC_ACCURACY_UPDATED(R37, B47, A47)` |
| S47 | `=CALC_ACCURACY_UPDATED(S37, B47, A47)` |
| C48 | `=CALC_ACCURACY_UPDATED(C37, B48, A48)` |
| D48 | `=CALC_ACCURACY_UPDATED(D37, B48,A48)` |
| E48 | `=CALC_ACCURACY_UPDATED(E37, B48, A48)` |
| F48 | `=CALC_ACCURACY_UPDATED(F37,B48,A48)` |
| G48 | `=CALC_ACCURACY_UPDATED(G37, B48, A48)` |
| H48 | `=CALC_ACCURACY_UPDATED(H37, B48, A48)` |
| I48 | `=CALC_ACCURACY_UPDATED(I37, B48,A48)` |
| J48 | `=CALC_ACCURACY_UPDATED(J37, B48,A48)` |
| K48 | `=CALC_ACCURACY_UPDATED(K37, B48,A48)` |
| M48 | `=CALC_ACCURACY_UPDATED(M37, B48,A48)` |
| N48 | `=CALC_ACCURACY_UPDATED(N37, B48,A48)` |
| O48 | `=CALC_ACCURACY_UPDATED(O37, B48,A48)` |
| Q48 | `=CALC_ACCURACY_UPDATED(Q37, B48, A48)` |
| R48 | `=CALC_ACCURACY_UPDATED(R37, B48, A48)` |
| S48 | `=CALC_ACCURACY_UPDATED(S37, B48, A48)` |
| C49 | `=CALC_ACCURACY_UPDATED(C37, B49, A49)` |
| D49 | `=CALC_ACCURACY_UPDATED(D37, B49,A49)` |
| E49 | `=CALC_ACCURACY_UPDATED(E37, B49, A49)` |
| F49 | `=CALC_ACCURACY_UPDATED(F37,B49,A49)` |
| G49 | `=CALC_ACCURACY_UPDATED(G37, B49, A49)` |
| H49 | `=CALC_ACCURACY_UPDATED(H37, B49, A49)` |
| I49 | `=CALC_ACCURACY_UPDATED(I37, B49,A49)` |
| J49 | `=CALC_ACCURACY_UPDATED(J37, B49,A49)` |
| K49 | `=CALC_ACCURACY_UPDATED(K37, B49,A49)` |
| M49 | `=CALC_ACCURACY_UPDATED(M37, B49,A49)` |
| N49 | `=CALC_ACCURACY_UPDATED(N37, B49,A49)` |
| O49 | `=CALC_ACCURACY_UPDATED(O37, B49,A49)` |
| Q49 | `=CALC_ACCURACY_UPDATED(Q37, B49, A49)` |
| R49 | `=CALC_ACCURACY_UPDATED(R37, B49, A49)` |
| S49 | `=CALC_ACCURACY_UPDATED(S37, B49, A49)` |
| C50 | `=CALC_ACCURACY_UPDATED(C37, B50, A50)` |
| D50 | `=CALC_ACCURACY_UPDATED(D37, B50,A50)` |
| E50 | `=CALC_ACCURACY_UPDATED(E37, B50, A50)` |
| F50 | `=CALC_ACCURACY_UPDATED(F37,B50,A50)` |
| G50 | `=CALC_ACCURACY_UPDATED(G37, B50, A50)` |
| H50 | `=CALC_ACCURACY_UPDATED(H37, B50, A50)` |
| I50 | `=CALC_ACCURACY_UPDATED(I37, B50,A50)` |
| J50 | `=CALC_ACCURACY_UPDATED(J37, B50,A50)` |
| K50 | `=CALC_ACCURACY_UPDATED(K37, B50,A50)` |
| M50 | `=CALC_ACCURACY_UPDATED(M37, B50,A50)` |
| N50 | `=CALC_ACCURACY_UPDATED(N37, B50,A50)` |
| O50 | `=CALC_ACCURACY_UPDATED(O37, B50,A50)` |
| Q50 | `=CALC_ACCURACY_UPDATED(Q37, B50, A50)` |
| R50 | `=CALC_ACCURACY_UPDATED(R37, B50, A50)` |
| S50 | `=CALC_ACCURACY_UPDATED(S37, B50, A50)` |
| C51 | `=CALC_ACCURACY_UPDATED(C37, B51, A51)` |
| D51 | `=CALC_ACCURACY_UPDATED(D37,B51,A51)` |
| E51 | `=CALC_ACCURACY_UPDATED(E37, B51, A51)` |
| F51 | `=CALC_ACCURACY_UPDATED(F37,B51,A51)` |
| G51 | `=CALC_ACCURACY_UPDATED(G37, B51, A51)` |
| H51 | `=CALC_ACCURACY_UPDATED(H37, B51, A51)` |
| I51 | `=CALC_ACCURACY_UPDATED(I37, B51,A51)` |
| J51 | `=CALC_ACCURACY_UPDATED(J37, B51,A51)` |
| K51 | `=CALC_ACCURACY_UPDATED(K37, B51,A51)` |
| M51 | `=CALC_ACCURACY_UPDATED(M37, B51,A51)` |
| N51 | `=CALC_ACCURACY_UPDATED(N37, B51,A51)` |
| O51 | `=CALC_ACCURACY_UPDATED(O37, B51,A51)` |
| Q51 | `=CALC_ACCURACY_UPDATED(Q37, B51, A51)` |
| R51 | `=CALC_ACCURACY_UPDATED(R37, B51, A51)` |
| S51 | `=CALC_ACCURACY_UPDATED(S37, B51, A51)` |
| C52 | `=CALC_ACCURACY_UPDATED(C37, B52, "Diagnosis")` |
| D52 | `=CALC_ACCURACY_UPDATED(D37, "", "Diagnosis")` |
| E52 | `=CALC_ACCURACY_UPDATED(E37, B52, "Diagnosis")` |
| F52 | `=CALC_ACCURACY(F37, "", "Diagnosis")` |
| G52 | `=CALC_ACCURACY_UPDATED(G37, B52, "Diagnosis")` |
| H52 | `=CALC_ACCURACY_UPDATED(H37, B52, "Diagnosis")` |
| I52 | `=CALC_ACCURACY_UPDATED(I37, B52, "Diagnosis")` |
| J52 | `=CALC_ACCURACY_UPDATED(J37, B52, "Diagnosis")` |
| K52 | `=CALC_ACCURACY_UPDATED(K37, B52, "Diagnosis")` |
| M52 | `=CALC_ACCURACY_UPDATED(M37, B52, "Diagnosis")` |
| N52 | `=CALC_ACCURACY_UPDATED(N37, B52, "Diagnosis")` |
| O52 | `=CALC_ACCURACY_UPDATED(O37, B52, "Diagnosis")` |
| Q52 | `=CALC_ACCURACY_UPDATED(Q37, B52, "Diagnosis")` |
| R52 | `=CALC_ACCURACY_UPDATED(R37, B52, "Diagnosis")` |
| S52 | `=CALC_ACCURACY_UPDATED(S37, B52, "Diagnosis")` |
| C53 | `=CALC_ACCURACY_UPDATED(C37, B53, "Management")` |
| D53 | `=CALC_ACCURACY_UPDATED(D37, "", "Management")` |
| E53 | `=CALC_ACCURACY_UPDATED(E37, B53, "Management")` |
| F53 | `=CALC_ACCURACY_UPDATED(F37, "", "Management")` |
| G53 | `=CALC_ACCURACY_UPDATED(G37, B53, "Management")` |
| H53 | `=CALC_ACCURACY_UPDATED(H37, B53, "Management")` |
| I53 | `=CALC_ACCURACY_UPDATED(I37, B53, "Management")` |
| J53 | `=CALC_ACCURACY_UPDATED(J37, B53, "Management")` |
| K53 | `=CALC_ACCURACY_UPDATED(K37, B53, "Management")` |
| M53 | `=CALC_ACCURACY_UPDATED(M37, B53, "Management")` |
| N53 | `=CALC_ACCURACY_UPDATED(N37, B53, "Management")` |
| O53 | `=CALC_ACCURACY_UPDATED(O37, B53, "Management")` |
| Q53 | `=CALC_ACCURACY_UPDATED(Q37, B53, "Management")` |
| R53 | `=CALC_ACCURACY_UPDATED(R37, B53, "Management")` |
| S53 | `=CALC_ACCURACY_UPDATED(S37, B53, "Management")` |
| C54 | `=COMBINE_DIAG_MANAG(C52, C53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| D54 | `=COMBINE_DIAG_MANAG(D52, D53, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| E54 | `=COMBINE_DIAG_MANAG(E52, E53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| F54 | `=COMBINE_DIAG_MANAG(F52, F53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| G54 | `=COMBINE_DIAG_MANAG(G52, G53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| H54 | `=COMBINE_DIAG_MANAG(H52, H53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| I54 | `=COMBINE_DIAG_MANAG(I52, I53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| J54 | `=COMBINE_DIAG_MANAG(J52, J53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| K54 | `=COMBINE_DIAG_MANAG(K52, K53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| M54 | `=COMBINE_DIAG_MANAG(M52, M53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| N54 | `=COMBINE_DIAG_MANAG(N52, N53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| O54 | `=COMBINE_DIAG_MANAG(O52, O53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| Q54 | `=COMBINE_DIAG_MANAG(Q52, Q53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| R54 | `=COMBINE_DIAG_MANAG(R52, R53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| S54 | `=COMBINE_DIAG_MANAG(S52, S53, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| C55 | `=CALC_ACCURACY_UPDATED(C37, "", "")` |
| D55 | `=CALC_ACCURACY_UPDATED(D37, "", "")` |
| E55 | `=CALC_ACCURACY_UPDATED(E37, "", "")` |
| F55 | `=CALC_ACCURACY_UPDATED(F37, "", "")` |
| G55 | `=CALC_ACCURACY_UPDATED(G37, "", "")` |
| H55 | `=CALC_ACCURACY_UPDATED(H37, "", "")` |
| I55 | `=CALC_ACCURACY_UPDATED(I37, "", "")` |
| J55 | `=CALC_ACCURACY_UPDATED(J37, "", "")` |
| K55 | `=CALC_ACCURACY_UPDATED(K37, "", "")` |
| M55 | `=CALC_ACCURACY_UPDATED(M37, "", "")` |
| N55 | `=CALC_ACCURACY_UPDATED(N37, "", "")` |
| O55 | `=CALC_ACCURACY_UPDATED(O37, "", "")` |
| Q55 | `=CALC_ACCURACY_UPDATED(Q37, "", "")` |
| R55 | `=CALC_ACCURACY_UPDATED(R37, "", "")` |
| S55 | `=CALC_ACCURACY_UPDATED(S37, "", "")` |
| C65 | `=CALC_ACCURACY_UPDATED(C62, B65, A65)` |
| D65 | `=CALC_ACCURACY_UPDATED(D62, B65,A65)` |
| E65 | `=CALC_ACCURACY_UPDATED(E62, B65, A65)` |
| F65 | `=CALC_ACCURACY_UPDATED(F62,B65,A65)` |
| G65 | `=CALC_ACCURACY_UPDATED(G62, B65, A65)` |
| H65 | `=CALC_ACCURACY_UPDATED(H62, B65, A65)` |
| I65 | `=CALC_ACCURACY_UPDATED(I62, B65,A65)` |
| J65 | `=CALC_ACCURACY_UPDATED(J62, B65,A65)` |
| K65 | `=CALC_ACCURACY_UPDATED(K62, B65,A65)` |
| L65 | `=MAX(I65:K65)` |
| M65 | `=CALC_ACCURACY_UPDATED(M62, B65,A65)` |
| N65 | `=CALC_ACCURACY_UPDATED(N62, B65,A65)` |
| O65 | `=CALC_ACCURACY_UPDATED(O62, B65,A65)` |
| P65 | `=MAX(M65:O65)` |
| Q65 | `=CALC_ACCURACY_UPDATED(Q62, B65, A65)` |
| R65 | `=CALC_ACCURACY_UPDATED(R62, B65, A65)` |
| S65 | `=CALC_ACCURACY_UPDATED(S62, B65, A65)` |
| C66 | `=CALC_ACCURACY_UPDATED(C62, B66, A66)` |
| D66 | `=CALC_ACCURACY_UPDATED(D62, B66,A66)` |
| E66 | `=CALC_ACCURACY_UPDATED(E62, B66, A66)` |
| F66 | `=CALC_ACCURACY_UPDATED(F62,B66,A66)` |
| G66 | `=CALC_ACCURACY_UPDATED(G62, B66, A66)` |
| H66 | `=CALC_ACCURACY_UPDATED(H62, B66, A66)` |
| I66 | `=CALC_ACCURACY_UPDATED(I62, B66,A66)` |
| J66 | `=CALC_ACCURACY_UPDATED(J62, B66,A66)` |
| K66 | `=CALC_ACCURACY_UPDATED(K62, B66,A66)` |
| L66 | `=MAX(I66:K66)` |
| M66 | `=CALC_ACCURACY_UPDATED(M62, B66,A66)` |
| N66 | `=CALC_ACCURACY_UPDATED(N62, B66,A66)` |
| O66 | `=CALC_ACCURACY_UPDATED(O62, B66,A66)` |
| Q66 | `=CALC_ACCURACY_UPDATED(Q62, B66, A66)` |
| R66 | `=CALC_ACCURACY_UPDATED(R62, B66, A66)` |
| S66 | `=CALC_ACCURACY_UPDATED(S62, B66, A66)` |
| C67 | `=CALC_ACCURACY_UPDATED(C62, B67, A67)` |
| D67 | `=CALC_ACCURACY_UPDATED(D62, B67,A67)` |
| E67 | `=CALC_ACCURACY_UPDATED(E62, B67, A67)` |
| F67 | `=CALC_ACCURACY_UPDATED(F62,B67,A67)` |
| G67 | `=CALC_ACCURACY_UPDATED(G62, B67, A67)` |
| H67 | `=CALC_ACCURACY_UPDATED(H62, B67, A67)` |
| I67 | `=CALC_ACCURACY_UPDATED(I62, B67,A67)` |
| J67 | `=CALC_ACCURACY_UPDATED(J62, B67,A67)` |
| K67 | `=CALC_ACCURACY_UPDATED(K62, B67,A67)` |
| L67 | `=MAX(I67:K67)` |
| M67 | `=CALC_ACCURACY_UPDATED(M62, B67,A67)` |
| N67 | `=CALC_ACCURACY_UPDATED(N62, B67,A67)` |
| O67 | `=CALC_ACCURACY_UPDATED(O62, B67,A67)` |
| Q67 | `=CALC_ACCURACY_UPDATED(Q62, B67, A67)` |
| R67 | `=CALC_ACCURACY_UPDATED(R62, B67, A67)` |
| S67 | `=CALC_ACCURACY_UPDATED(S62, B67, A67)` |
| C68 | `=CALC_ACCURACY_UPDATED(C62, B68, A68)` |
| D68 | `=CALC_ACCURACY_UPDATED(D62, B68,A68)` |
| E68 | `=CALC_ACCURACY_UPDATED(E62, B68, A68)` |
| F68 | `=CALC_ACCURACY_UPDATED(F62,B68,A68)` |
| G68 | `=CALC_ACCURACY_UPDATED(G62, B68, A68)` |
| H68 | `=CALC_ACCURACY_UPDATED(H62, B68, A68)` |
| I68 | `=CALC_ACCURACY_UPDATED(I62, B68,A68)` |
| J68 | `=CALC_ACCURACY_UPDATED(J62, B68,A68)` |
| K68 | `=CALC_ACCURACY_UPDATED(K62, B68,A68)` |
| L68 | `=MAX(I68:K68)` |
| M68 | `=CALC_ACCURACY_UPDATED(M62, B68,A68)` |
| N68 | `=CALC_ACCURACY_UPDATED(N62, B68,A68)` |
| O68 | `=CALC_ACCURACY_UPDATED(O62, B68,A68)` |
| Q68 | `=CALC_ACCURACY_UPDATED(Q62, B68, A68)` |
| R68 | `=CALC_ACCURACY_UPDATED(R62, B68, A68)` |
| S68 | `=CALC_ACCURACY_UPDATED(S62, B68, A68)` |
| C69 | `=CALC_ACCURACY_UPDATED(C62, B69, A69)` |
| D69 | `=CALC_ACCURACY_UPDATED(D62, B69,A69)` |
| E69 | `=CALC_ACCURACY_UPDATED(E62, B69, A69)` |
| F69 | `=CALC_ACCURACY_UPDATED(F62,B69,A69)` |
| G69 | `=CALC_ACCURACY_UPDATED(G62, B69, A69)` |
| H69 | `=CALC_ACCURACY_UPDATED(H62, B69, A69)` |
| I69 | `=CALC_ACCURACY_UPDATED(I62, B69,A69)` |
| J69 | `=CALC_ACCURACY_UPDATED(J62, B69,A69)` |
| K69 | `=CALC_ACCURACY_UPDATED(K62, B69,A69)` |
| L69 | `=MAX(I69:K69)` |
| M69 | `=CALC_ACCURACY_UPDATED(M62, B69,A69)` |
| N69 | `=CALC_ACCURACY_UPDATED(N62, B69,A69)` |
| O69 | `=CALC_ACCURACY_UPDATED(O62, B69,A69)` |
| Q69 | `=CALC_ACCURACY_UPDATED(Q62, B69, A69)` |
| R69 | `=CALC_ACCURACY_UPDATED(R62, B69, A69)` |
| S69 | `=CALC_ACCURACY_UPDATED(S62, B69, A69)` |
| C70 | `=CALC_ACCURACY_UPDATED(C62, B70, A70)` |
| D70 | `=CALC_ACCURACY_UPDATED(D62, B70,A70)` |
| E70 | `=CALC_ACCURACY_UPDATED(E62, B70, A70)` |
| F70 | `=CALC_ACCURACY_UPDATED(F62,B70,A70)` |
| G70 | `=CALC_ACCURACY_UPDATED(G62, B70, A70)` |
| H70 | `=CALC_ACCURACY_UPDATED(H62, B70, A70)` |
| I70 | `=CALC_ACCURACY_UPDATED(I62, B70,A70)` |
| J70 | `=CALC_ACCURACY_UPDATED(J62, B70,A70)` |
| K70 | `=CALC_ACCURACY_UPDATED(K62, B70,A70)` |
| L70 | `=MAX(I70:K70)` |
| M70 | `=CALC_ACCURACY_UPDATED(M62, B70,A70)` |
| N70 | `=CALC_ACCURACY_UPDATED(N62, B70,A70)` |
| O70 | `=CALC_ACCURACY_UPDATED(O62, B70,A70)` |
| Q70 | `=CALC_ACCURACY_UPDATED(Q62, B70, A70)` |
| R70 | `=CALC_ACCURACY_UPDATED(R62, B70, A70)` |
| S70 | `=CALC_ACCURACY_UPDATED(S62, B70, A70)` |
| C71 | `=CALC_ACCURACY_UPDATED(C62, B71, A71)` |
| D71 | `=CALC_ACCURACY_UPDATED(D62, B71,A71)` |
| E71 | `=CALC_ACCURACY_UPDATED(E62, B71, A71)` |
| F71 | `=CALC_ACCURACY_UPDATED(F62,B71,A71)` |
| G71 | `=CALC_ACCURACY_UPDATED(G62, B71, A71)` |
| H71 | `=CALC_ACCURACY_UPDATED(H62, B71, A71)` |
| I71 | `=CALC_ACCURACY_UPDATED(I62, B71,A71)` |
| J71 | `=CALC_ACCURACY_UPDATED(J62, B71,A71)` |
| K71 | `=CALC_ACCURACY_UPDATED(K62, B71,A71)` |
| L71 | `=MAX(I71:K71)` |
| M71 | `=CALC_ACCURACY_UPDATED(M62, B71,A71)` |
| N71 | `=CALC_ACCURACY_UPDATED(N62, B71,A71)` |
| O71 | `=CALC_ACCURACY_UPDATED(O62, B71,A71)` |
| Q71 | `=CALC_ACCURACY_UPDATED(Q62, B71, A71)` |
| R71 | `=CALC_ACCURACY_UPDATED(R62, B71, A71)` |
| S71 | `=CALC_ACCURACY_UPDATED(S62, B71, A71)` |
| C72 | `=CALC_ACCURACY_UPDATED(C62, B72, A72)` |
| D72 | `=CALC_ACCURACY_UPDATED(D62, B72,A72)` |
| E72 | `=CALC_ACCURACY_UPDATED(E62, B72, A72)` |
| F72 | `=CALC_ACCURACY_UPDATED(F62,B72,A72)` |
| G72 | `=CALC_ACCURACY_UPDATED(G62, B72, A72)` |
| H72 | `=CALC_ACCURACY_UPDATED(H62, B72, A72)` |
| I72 | `=CALC_ACCURACY_UPDATED(I62, B72,A72)` |
| J72 | `=CALC_ACCURACY_UPDATED(J62, B72,A72)` |
| K72 | `=CALC_ACCURACY_UPDATED(K62, B72,A72)` |
| L72 | `=MAX(I72:K72)` |
| M72 | `=CALC_ACCURACY_UPDATED(M62, B72,A72)` |
| N72 | `=CALC_ACCURACY_UPDATED(N62, B72,A72)` |
| O72 | `=CALC_ACCURACY_UPDATED(O62, B72,A72)` |
| Q72 | `=CALC_ACCURACY_UPDATED(Q62, B72, A72)` |
| R72 | `=CALC_ACCURACY_UPDATED(R62, B72, A72)` |
| S72 | `=CALC_ACCURACY_UPDATED(S62, B72, A72)` |
| C73 | `=CALC_ACCURACY_UPDATED(C62, B73, A73)` |
| D73 | `=CALC_ACCURACY_UPDATED(D62, B73,A73)` |
| E73 | `=CALC_ACCURACY_UPDATED(E62, B73, A73)` |
| F73 | `=CALC_ACCURACY_UPDATED(F62,B73,A73)` |
| G73 | `=CALC_ACCURACY_UPDATED(G62, B73, A73)` |
| H73 | `=CALC_ACCURACY_UPDATED(H62, B73, A73)` |
| I73 | `=CALC_ACCURACY_UPDATED(I62, B73,A73)` |
| J73 | `=CALC_ACCURACY_UPDATED(J62, B73,A73)` |
| K73 | `=CALC_ACCURACY_UPDATED(K62, B73,A73)` |
| L73 | `=MAX(I73:K73)` |
| M73 | `=CALC_ACCURACY_UPDATED(M62, B73,A73)` |
| N73 | `=CALC_ACCURACY_UPDATED(N62, B73,A73)` |
| O73 | `=CALC_ACCURACY_UPDATED(O62, B73,A73)` |
| Q73 | `=CALC_ACCURACY_UPDATED(Q62, B73, A73)` |
| R73 | `=CALC_ACCURACY_UPDATED(R62, B73, A73)` |
| S73 | `=CALC_ACCURACY_UPDATED(S62, B73, A73)` |
| C74 | `=CALC_ACCURACY_UPDATED(C62, B74, A74)` |
| D74 | `=CALC_ACCURACY_UPDATED(D62, B74,A74)` |
| E74 | `=CALC_ACCURACY_UPDATED(E62, B74, A74)` |
| F74 | `=CALC_ACCURACY_UPDATED(F62,B74,A74)` |
| G74 | `=CALC_ACCURACY_UPDATED(G62, B74, A74)` |
| H74 | `=CALC_ACCURACY_UPDATED(H62, B74, A74)` |
| I74 | `=CALC_ACCURACY_UPDATED(I62, B74,A74)` |
| J74 | `=CALC_ACCURACY_UPDATED(J62, B74,A74)` |
| K74 | `=CALC_ACCURACY_UPDATED(K62, B74,A74)` |
| L74 | `=MAX(I74:K74)` |
| M74 | `=CALC_ACCURACY_UPDATED(M62, B74,A74)` |
| N74 | `=CALC_ACCURACY_UPDATED(N62, B74,A74)` |
| O74 | `=CALC_ACCURACY_UPDATED(O62, B74,A74)` |
| Q74 | `=CALC_ACCURACY_UPDATED(Q62, B74, A74)` |
| R74 | `=CALC_ACCURACY_UPDATED(R62, B74, A74)` |
| S74 | `=CALC_ACCURACY_UPDATED(S62, B74, A74)` |
| C75 | `=CALC_ACCURACY_UPDATED(C62, B75, A75)` |
| D75 | `=CALC_ACCURACY_UPDATED(D62, B75,A75)` |
| E75 | `=CALC_ACCURACY_UPDATED(E62, B75, A75)` |
| F75 | `=CALC_ACCURACY_UPDATED(F62,B75,A75)` |
| G75 | `=CALC_ACCURACY_UPDATED(G62, B75, A75)` |
| H75 | `=CALC_ACCURACY_UPDATED(H62, B75, A75)` |
| I75 | `=CALC_ACCURACY_UPDATED(I62, B75,A75)` |
| J75 | `=CALC_ACCURACY_UPDATED(J62, B75,A75)` |
| K75 | `=CALC_ACCURACY_UPDATED(K62, B75,A75)` |
| L75 | `=MAX(I75:K75)` |
| M75 | `=CALC_ACCURACY_UPDATED(M62, B75,A75)` |
| N75 | `=CALC_ACCURACY_UPDATED(N62, B75,A75)` |
| O75 | `=CALC_ACCURACY_UPDATED(O62, B75,A75)` |
| Q75 | `=CALC_ACCURACY_UPDATED(Q62, B75, A75)` |
| R75 | `=CALC_ACCURACY_UPDATED(R62, B75, A75)` |
| S75 | `=CALC_ACCURACY_UPDATED(S62, B75, A75)` |
| C76 | `=CALC_ACCURACY_UPDATED(C62, B76, A76)` |
| D76 | `=CALC_ACCURACY_UPDATED(D62,B76,A76)` |
| E76 | `=CALC_ACCURACY_UPDATED(E62, B76, A76)` |
| F76 | `=CALC_ACCURACY_UPDATED(F62,B76,A76)` |
| G76 | `=CALC_ACCURACY_UPDATED(G62, B76, A76)` |
| H76 | `=CALC_ACCURACY_UPDATED(H62, B76, A76)` |
| I76 | `=CALC_ACCURACY_UPDATED(I62, B76,A76)` |
| J76 | `=CALC_ACCURACY_UPDATED(J62, B76,A76)` |
| K76 | `=CALC_ACCURACY_UPDATED(K62, B76,A76)` |
| L76 | `=MAX(I76:K76)` |
| M76 | `=CALC_ACCURACY_UPDATED(M62, B76,A76)` |
| N76 | `=CALC_ACCURACY_UPDATED(N62, B76,A76)` |
| O76 | `=CALC_ACCURACY_UPDATED(O62, B76,A76)` |
| Q76 | `=CALC_ACCURACY_UPDATED(Q62, B76, A76)` |
| R76 | `=CALC_ACCURACY_UPDATED(R62, B76, A76)` |
| S76 | `=CALC_ACCURACY_UPDATED(S62, B76, A76)` |
| C77 | `=CALC_ACCURACY_UPDATED(C62, B77, "Diagnosis")` |
| D77 | `=CALC_ACCURACY_UPDATED(D62, "", "Diagnosis")` |
| E77 | `=CALC_ACCURACY_UPDATED(E62, B77, "Diagnosis")` |
| F77 | `=CALC_ACCURACY(F62, "", "Diagnosis")` |
| G77 | `=CALC_ACCURACY_UPDATED(G62, B77, "Diagnosis")` |
| H77 | `=CALC_ACCURACY_UPDATED(H62, B77, "Diagnosis")` |
| I77 | `=CALC_ACCURACY_UPDATED(I62, B77, "Diagnosis")` |
| J77 | `=CALC_ACCURACY_UPDATED(J62, B77, "Diagnosis")` |
| K77 | `=CALC_ACCURACY_UPDATED(K62, B77, "Diagnosis")` |
| L77 | `=MAX(I77:K77)` |
| M77 | `=CALC_ACCURACY_UPDATED(M62, B77, "Diagnosis")` |
| N77 | `=CALC_ACCURACY_UPDATED(N62, B77, "Diagnosis")` |
| O77 | `=CALC_ACCURACY_UPDATED(O62, B77, "Diagnosis")` |
| Q77 | `=CALC_ACCURACY_UPDATED(Q62, B77, "Diagnosis")` |
| R77 | `=CALC_ACCURACY_UPDATED(R62, B77, "Diagnosis")` |
| S77 | `=CALC_ACCURACY_UPDATED(S62, B77, "Diagnosis")` |
| C78 | `=CALC_ACCURACY_UPDATED(C62, B78, "Management")` |
| D78 | `=CALC_ACCURACY_UPDATED(D62, "", "Management")` |
| E78 | `=CALC_ACCURACY_UPDATED(E62, B78, "Management")` |
| F78 | `=CALC_ACCURACY_UPDATED(F62, "", "Management")` |
| G78 | `=CALC_ACCURACY_UPDATED(G62, B78, "Management")` |
| H78 | `=CALC_ACCURACY_UPDATED(H62, B78, "Management")` |
| I78 | `=CALC_ACCURACY_UPDATED(I62, B78, "Management")` |
| J78 | `=CALC_ACCURACY_UPDATED(J62, B78, "Management")` |
| K78 | `=CALC_ACCURACY_UPDATED(K62, B78, "Management")` |
| L78 | `=MAX(I78:K78)` |
| M78 | `=CALC_ACCURACY_UPDATED(M62, B78, "Management")` |
| N78 | `=CALC_ACCURACY_UPDATED(N62, B78, "Management")` |
| O78 | `=CALC_ACCURACY_UPDATED(O62, B78, "Management")` |
| Q78 | `=CALC_ACCURACY_UPDATED(Q62, B78, "Management")` |
| R78 | `=CALC_ACCURACY_UPDATED(R62, B78, "Management")` |
| S78 | `=CALC_ACCURACY_UPDATED(S62, B78, "Management")` |
| C79 | `=COMBINE_DIAG_MANAG(C77, C78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| D79 | `=COMBINE_DIAG_MANAG(D77, D78, COUNTIF(Table3[Type],"Diagnosis"), COUNTIF(Table3[Type],"Management"))` |
| E79 | `=COMBINE_DIAG_MANAG(E77, E78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| F79 | `=COMBINE_DIAG_MANAG(F77, F78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| G79 | `=COMBINE_DIAG_MANAG(G77, G78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| H79 | `=COMBINE_DIAG_MANAG(H77, H78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| I79 | `=COMBINE_DIAG_MANAG(I77, I78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| J79 | `=COMBINE_DIAG_MANAG(J77, J78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| K79 | `=COMBINE_DIAG_MANAG(K77, K78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| L79 | `=MAX(I79:K79)` |
| M79 | `=COMBINE_DIAG_MANAG(M77, M78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| N79 | `=COMBINE_DIAG_MANAG(N77, N78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| O79 | `=COMBINE_DIAG_MANAG(O77, O78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| Q79 | `=COMBINE_DIAG_MANAG(Q77, Q78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| R79 | `=COMBINE_DIAG_MANAG(R77, R78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| S79 | `=COMBINE_DIAG_MANAG(S77, S78, COUNTIF(Ophthalmology!$C$2:$C$131,"Diagnosis"), COUNTIF(Ophthalmology!$C$2:$C$131,"Management"))` |
| C80 | `=CALC_ACCURACY_UPDATED(C62, "", "")` |
| D80 | `=CALC_ACCURACY_UPDATED(D62, "", "")` |
| E80 | `=CALC_ACCURACY_UPDATED(E62, "", "")` |
| F80 | `=CALC_ACCURACY_UPDATED(F62, "", "")` |
| G80 | `=CALC_ACCURACY_UPDATED(G62, "", "")` |
| H80 | `=CALC_ACCURACY_UPDATED(H62, "", "")` |
| I80 | `=CALC_ACCURACY_UPDATED(I62, "", "")` |
| J80 | `=CALC_ACCURACY_UPDATED(J62, "", "")` |
| K80 | `=CALC_ACCURACY_UPDATED(K62, "", "")` |
| L80 | `=MAX(I80:K80)` |
| M80 | `=CALC_ACCURACY_UPDATED(M62, "", "")` |
| N80 | `=CALC_ACCURACY_UPDATED(N62, "", "")` |
| O80 | `=CALC_ACCURACY_UPDATED(O62, "", "")` |
| Q80 | `=CALC_ACCURACY_UPDATED(Q62, "", "")` |
| R80 | `=CALC_ACCURACY_UPDATED(R62, "", "")` |
| S80 | `=CALC_ACCURACY_UPDATED(S62, "", "")` |

### Ophthalmology (82 formulas)

| Cell | Formula |
|------|--------|
| F133 | `=ACCURACY(   INDEX(2:131, , MATCH("GT", 1:1, 0)),   F2:F131 ) ` |
| G133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  G2:G131) ` |
| H133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  H2:H131) ` |
| I133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  I2:I131) ` |
| J133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  J2:J131) ` |
| K133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  K2:K131) ` |
| L133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  L2:L131) ` |
| M133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  M2:M131) ` |
| N133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  N2:N131) ` |
| O133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  O2:O131) ` |
| P133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  P2:P131) ` |
| Q133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  Q2:Q131) ` |
| R133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  R2:R131) ` |
| S133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  S2:S131) ` |
| T133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  T2:T131) ` |
| U133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  U2:U131) ` |
| V133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  V2:V131) ` |
| W133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  W2:W131) ` |
| X133 | `=SET_ACCURACY(   INDEX(2:131, , MATCH("GT", 1:1, 0)),   X2:X131 ) ` |
| Y133 | `=SET_ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  Y2:Y131) ` |
| Z133 | `=SET_ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  Z2:Z131) ` |
| AA133 | `=ACCURACY(   INDEX(2:131, , MATCH("GT", 1:1, 0)),   AA2:AA131 ) ` |
| AB133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AB2:AB131) ` |
| AC133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AC2:AC131) ` |
| AD133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AD2:AD131) ` |
| AE133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AE2:AE131) ` |
| AF133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AF2:AF131) ` |
| AG133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AG2:AG131) ` |
| AH133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AH2:AH131) ` |
| AI133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AI2:AI131) ` |
| AJ133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AJ2:AJ131) ` |
| AK133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AK2:AK131) ` |
| AL133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AL2:AL131) ` |
| AM133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AM2:AM131) ` |
| AN133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AN2:AN131) ` |
| AO133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AO2:AO131) ` |
| AP133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AP2:AP131) ` |
| AQ133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AQ2:AQ131) ` |
| AR133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AR2:AR131) ` |
| AS133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AS2:AS131) ` |
| AT133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AT2:AT131) ` |
| AU133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AU2:AU131) ` |
| AV133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AV2:AV131) ` |
| AW133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AW2:AW131) ` |
| AX133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AX2:AX131) ` |
| AY133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AY2:AY131) ` |
| AZ133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  AZ2:AZ131) ` |
| BA133 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  BA2:BA131) ` |
| X134 | `=ACCURACY(   INDEX(2:131, , MATCH("GT", 1:1, 0)),   X2:X131 ) ` |
| Y134 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  Y2:Y131) ` |
| Z134 | `=ACCURACY( INDEX(2:131, , MATCH("GT", 1:1, 0)),  Z2:Z131) ` |
| F136 | `=MCQ_CI_OPTIONB(Table3[GT], Table3[gpt5-0807-m1], Table3[gpt5-0807-m2], Table3[gpt5-0807-m3]) ` |
| F137 | `=SC_ACCURACY_CI(Table3[GT], Table3[gpt5-0807-m1], Table3[gpt5-0807-m2], Table3[gpt5-0807-m3], TRUE) ` |
| R137 | `=SC_ACCURACY_CI(Table3[GT], Table3[o4-mini-m1], Table3[o4-mini-m2], Table3[o4-mini-m3], TRUE) ` |
| X137 | `=SC_ACCURACY_CI(Table3[GT], Table3[ds-r1-0528-v1], Table3[ds-r1-0528-v2], Table3[ds-r1-0528-v3], TRUE) ` |
| AA137 | `=SC_ACCURACY_CI(Table3[GT], Table3[oss20b (L) v1], Table3[oss20b (L) v2], Table3[oss20b (L) v3], TRUE) ` |
| AD137 | `=SC_ACCURACY_CI(Table3[GT], Table3[oss20b (M) v1], Table3[oss20b (M) v2], Table3[oss20b (M) v3], TRUE) ` |
| AG137 | `=SC_ACCURACY_CI(Table3[GT], Table3[oss20b (H) v1], Table3[oss20b (H) v2], Table3[oss20b (H) v3], TRUE) ` |
| AJ137 | `=SC_ACCURACY_CI(Table3[GT], Table3[oss120b (L) v1], Table3[oss120b (L) v2], Table3[oss120b (L) v3], TRUE) ` |
| AM137 | `=SC_ACCURACY_CI(Table3[GT], Table3[oss120b (M) v1], Table3[oss120b (M) v2], Table3[oss120b (M) v3], TRUE) ` |
| AP137 | `=SC_ACCURACY_CI(Table3[GT], Table3[oss120b (H) v1], Table3[oss120b (H) v2], Table3[oss120b (H) v3], TRUE) ` |
| AS137 | `=SC_ACCURACY_CI(Table3[GT], Table3[qwen3.5 35B v1], Table3[qwen3.5 35B v2], Table3[qwen3.5 35B v3], TRUE) ` |
| F139 | `=MCQ_CI_TEXT(Table3[GT], Table3[gpt5-0807-m1], Table3[gpt5-0807-m2], Table3[gpt5-0807-m3]) ` |
| R139 | `=MCQ_CI_TEXT(Table3[GT], Table3[o4-mini-m1], Table3[o4-mini-m2], Table3[o4-mini-m3]) ` |
| X139 | `=MCQ_CI_TEXT(Table3[GT], Table3[ds-r1-0528-v1], Table3[ds-r1-0528-v2], Table3[ds-r1-0528-v3]) ` |
| AA139 | `=MCQ_CI_TEXT(Table3[GT], Table3[oss20b (L) v1], Table3[oss20b (L) v2], Table3[oss20b (L) v3]) ` |
| AD139 | `=MCQ_CI_TEXT(Table3[GT], Table3[oss20b (M) v1], Table3[oss20b (M) v2], Table3[oss20b (M) v3]) ` |
| AG139 | `=MCQ_CI_TEXT(Table3[GT], Table3[oss20b (H) v1], Table3[oss20b (H) v2], Table3[oss20b (H) v3]) ` |
| AJ139 | `=MCQ_CI_TEXT(Table3[GT], Table3[oss120b (L) v1], Table3[oss120b (L) v2], Table3[oss120b (L) v3]) ` |
| AM139 | `=MCQ_CI_TEXT(Table3[GT], Table3[oss120b (M) v1], Table3[oss120b (M) v2], Table3[oss120b (M) v3]) ` |
| AP139 | `=MCQ_CI_TEXT(Table3[GT], Table3[oss120b (H) v1], Table3[oss120b (H) v2], Table3[oss120b (H) v3]) ` |
| F141 | `=AVERAGE(F133:H133)` |
| R141 | `=AVERAGE(R133:T133)` |
| X141 | `=AVERAGE(X133:Z133)` |
| AA141 | `=AVERAGE(AA133:AC133)` |
| AD141 | `=AVERAGE(AD133:AF133)` |
| AG141 | `=AVERAGE(AG133:AI133)` |
| AJ141 | `=AVERAGE(AJ133:AL133)` |
| AM141 | `=AVERAGE(AM133:AO133)` |
| AP141 | `=AVERAGE(AP133:AR133)` |
| F142 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   INDEX(Ophthalmology!$2:$131, , MATCH(""GT"", Ophthalmology!$1:$1, 0)),    TRANSPOSE(     FILTER(       TRANSPOSE(Ophthalmology!$2:$131),       REGE...` |
| F143 | `=MCNEMAR_TEST(E2:E131, F2:F131, X2:X131)` |

### Copy of Eurorad (47 formulas)

| Cell | Formula |
|------|--------|
| H210 | `=MODEL_ACCURACY(   INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),   H2:H208 ) ` |
| I210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  I2:I208) ` |
| J210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  J2:J208) ` |
| K210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  K2:K208) ` |
| L210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  L2:L208) ` |
| M210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  M2:M208) ` |
| N210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  N2:N208) ` |
| O210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  O2:O208) ` |
| P210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  P2:P208) ` |
| Q210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  Q2:Q208) ` |
| R210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  R2:R208) ` |
| S210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  S2:S208) ` |
| T210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  T2:T208) ` |
| U210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  U2:U208) ` |
| V210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  V2:V208) ` |
| W210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  W2:W208) ` |
| X210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  X2:X208) ` |
| Y210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  Y2:Y208) ` |
| Z210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  Z2:Z208) ` |
| AA210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AA2:AA208) ` |
| AB210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AB2:AB208) ` |
| AC210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AC2:AC208) ` |
| AD210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AD2:AD208) ` |
| AE210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AE2:AE208) ` |
| AF210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AF2:AF208) ` |
| AG210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AG2:AG208) ` |
| AH210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AH2:AH208) ` |
| AI210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AI2:AI208) ` |
| AJ210 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  AJ2:AJ208) ` |
| H212 | `=MCQ_CI_OPTIONB($F$2:$F$208, H2:H208, I2:I208, J2:J208) ` |
| H213 | `=MCQ_CI_TEXT($F$2:$F$208, H2:H208, I2:I208, J2:J208)` |
| K213 | `=MCQ_CI_TEXT($F$2:$F$208, K2:K208, L2:L208, M2:M208)` |
| N213 | `=MCQ_CI_TEXT($F$2:$F$208, N2:N208, O2:O208, P2:P208)` |
| Q213 | `=MCQ_CI_TEXT($F$2:$F$208, Q2:Q208, R2:R208, S2:S208)` |
| T213 | `=MCQ_CI_TEXT($F$2:$F$208, T2:T208, U2:U208, V2:V208)` |
| W213 | `=MCQ_CI_TEXT($F$2:$F$208, W2:W208, X2:X208, Y2:Y208)` |
| X213 | `=MCQ_CI_TEXT($F$2:$F$208, X2:X208, Y2:Y208, Z2:Z208)` |
| AA213 | `=MCQ_CI_TEXT($F$2:$F$208, AA2:AA208, AB2:AB208, AC2:AC208)` |
| AB213 | `=MCQ_CI_TEXT($F$2:$F$208, AB2:AB208, AC2:AC208, AD2:AD208)` |
| AE213 | `=MCQ_CI_TEXT($F$2:$F$208, AE2:AE208, AF2:AF208, AG2:AG208)` |
| AH213 | `=MCQ_CI_TEXT($F$2:$F$208, AH2:AH208, AI2:AI208, AJ2:AJ208)` |
| H215 | `=MCNEMAR_VOTE_TEST('Copy of Eurorad'!F2:F208, 'Copy of Eurorad'!H2:J208, 'Copy of Eurorad'!K2:M208)` |
| K215 | `=MCNEMAR_VOTE_TEST('Copy of Eurorad'!F2:F208, 'Copy of Eurorad'!H2:J208, 'Copy of Eurorad'!N2:P208)` |
| H216 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   'Copy of Eurorad'!F2:F208,   INDEX(     'Copy of Eurorad'!2:208,     ,     FILTER(       COLUMN('Copy of Eurorad'!1:1),       REGEXMATCH('Copy of E...` |
| H217 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   'Copy of Eurorad'!F2:F208,   TRANSPOSE(     FILTER(       TRANSPOSE('Copy of Eurorad'!H2:M208),       REGEXMATCH(TRANSPOSE('Copy of Eurorad'!H1:M1)...` |
| K217 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   'Copy of Eurorad'!F2:F208,   TRANSPOSE(     FILTER(       TRANSPOSE('Copy of Eurorad'!H2:AJ208),       REGEXMATCH(TRANSPOSE('Copy of Eurorad'!H1:M1...` |
| K218 | `=IFERROR(__xludf.DUMMYFUNCTION("MCNEMAR_VOTE_TEST(   'Copy of Eurorad'!F2:F208,   TRANSPOSE(     FILTER(       TRANSPOSE('Copy of Eurorad'!H2:AJ208),       REGEXMATCH(TRANSPOSE('Copy of Eurorad'!H1:AJ...` |

### NMED Treatment (43 formulas)

| Cell | Formula |
|------|--------|
| G598 | `=MAE(   INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),   G2:G596 )` |
| H598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  H2:H596)` |
| I598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  I2:I596)` |
| L598 | `=MAE(   INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),   L2:L596 )` |
| M598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  M2:M596)` |
| N598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  N2:N596)` |
| O598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  O2:O596)` |
| G600 | `=MAE_IQR_TEXT($F$2:$F$596, G2:G596) ` |
| H600 | `=MAE_IQR_TEXT($F$2:$F$596, H2:H596) ` |
| I600 | `=MAE_IQR_TEXT($F$2:$F$596, I2:I596) ` |
| L600 | `=MAE_IQR_TEXT($F$2:$F$596, L2:L596) ` |
| M600 | `=MAE_IQR_TEXT($F$2:$F$596, M2:M596) ` |
| N600 | `=MAE_IQR_TEXT($F$2:$F$596, N2:N596) ` |
| O600 | `=MAE_IQR_TEXT($F$2:$F$596, O2:O596) ` |
| G601 | `=MEDAE_IQR_TEXT($F$2:$F$596, G2:G596)` |
| H601 | `=MEDAE_IQR_TEXT($F$2:$F$596, H2:H596)` |
| I601 | `=MEDAE_IQR_TEXT($F$2:$F$596, I2:I596)` |
| L601 | `=MEDAE_IQR_TEXT($F$2:$F$596, L2:L596)` |
| M601 | `=MEDAE_IQR_TEXT($F$2:$F$596, M2:M596)` |
| N601 | `=MEDAE_IQR_TEXT($F$2:$F$596, N2:N596)` |
| O601 | `=MEDAE_IQR_TEXT($F$2:$F$596, O2:O596)` |
| G602 | `=MEAN_SCORE_CI_TEXT(G2:G596)` |
| H602 | `=MEAN_SCORE_CI_TEXT(H2:H596)` |
| I602 | `=MEAN_SCORE_CI_TEXT(I2:I596)` |
| L602 | `=MEAN_SCORE_CI_TEXT(L2:L596)` |
| M602 | `=MEAN_SCORE_CI_TEXT(M2:M596)` |
| N602 | `=MEAN_SCORE_CI_TEXT(N2:N596)` |
| O602 | `=MEAN_SCORE_CI_TEXT(O2:O596)` |
| G603 | `=MAE_CI_TEXT($F$2:$F$596, G2:G596) ` |
| H603 | `=MAE_CI_TEXT($F$2:$F$596, H2:H596) ` |
| I603 | `=MAE_CI_TEXT($F$2:$F$596, I2:I596) ` |
| L603 | `=MAE_CI_TEXT($F$2:$F$596, L2:L596) ` |
| M603 | `=MAE_CI_TEXT($F$2:$F$596, M2:M596) ` |
| N603 | `=MAE_CI_TEXT($F$2:$F$596, N2:N596) ` |
| O603 | `=MAE_CI_TEXT($F$2:$F$596, O2:O596) ` |
| G605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, G2:G596)` |
| H605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, H2:H596)` |
| I605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, I2:I596)` |
| L605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, L2:L596)` |
| M605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, M2:M596)` |
| N605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, N2:N596)` |
| O605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, O2:O596)` |
| G609 | `=WILCOXON_ERROR_TEST(F2:F596, G2:G596, H2:H596)` |

### Copy of NMED Treatment (72 formulas)

| Cell | Formula |
|------|--------|
| G598 | `=MAE(   INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),   G2:G596 )` |
| H598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  H2:H596)` |
| I598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  I2:I596)` |
| J598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  J2:J596)` |
| K598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  K2:K596)` |
| L598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  L2:L596)` |
| M598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  M2:M596)` |
| N598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  N2:N596)` |
| O598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  O2:O596)` |
| P598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  P2:P596)` |
| Q598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  Q2:Q596)` |
| R598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  R2:R596)` |
| G600 | `=MAE_IQR_TEXT($F$2:$F$596, G2:G596) ` |
| H600 | `=MAE_IQR_TEXT($F$2:$F$596, H2:H596) ` |
| I600 | `=MAE_IQR_TEXT($F$2:$F$596, I2:I596) ` |
| J600 | `=MAE_IQR_TEXT($F$2:$F$596, J2:J596) ` |
| K600 | `=MAE_IQR_TEXT($F$2:$F$596, K2:K596) ` |
| L600 | `=MAE_IQR_TEXT($F$2:$F$596, L2:L596) ` |
| M600 | `=MAE_IQR_TEXT($F$2:$F$596, M2:M596) ` |
| N600 | `=MAE_IQR_TEXT($F$2:$F$596, N2:N596) ` |
| O600 | `=MAE_IQR_TEXT($F$2:$F$596, O2:O596) ` |
| P600 | `=MAE_IQR_TEXT($F$2:$F$596, P2:P596) ` |
| Q600 | `=MAE_IQR_TEXT($F$2:$F$596, Q2:Q596) ` |
| R600 | `=MAE_IQR_TEXT($F$2:$F$596, R2:R596) ` |
| G601 | `=MEDAE_IQR_TEXT($F$2:$F$596, G2:G596)` |
| H601 | `=MEDAE_IQR_TEXT($F$2:$F$596, H2:H596)` |
| I601 | `=MEDAE_IQR_TEXT($F$2:$F$596, I2:I596)` |
| J601 | `=MEDAE_IQR_TEXT($F$2:$F$596, J2:J596)` |
| K601 | `=MEDAE_IQR_TEXT($F$2:$F$596, K2:K596)` |
| L601 | `=MEDAE_IQR_TEXT($F$2:$F$596, L2:L596)` |
| M601 | `=MEDAE_IQR_TEXT($F$2:$F$596, M2:M596)` |
| N601 | `=MEDAE_IQR_TEXT($F$2:$F$596, N2:N596)` |
| O601 | `=MEDAE_IQR_TEXT($F$2:$F$596, O2:O596)` |
| P601 | `=MEDAE_IQR_TEXT($F$2:$F$596, P2:P596)` |
| Q601 | `=MEDAE_IQR_TEXT($F$2:$F$596, Q2:Q596)` |
| R601 | `=MEDAE_IQR_TEXT($F$2:$F$596, R2:R596)` |
| G602 | `=MEAN_SCORE_CI_TEXT(G2:G596)` |
| H602 | `=MEAN_SCORE_CI_TEXT(H2:H596)` |
| I602 | `=MEAN_SCORE_CI_TEXT(I2:I596)` |
| J602 | `=MEAN_SCORE_CI_TEXT(J2:J596)` |
| K602 | `=MEAN_SCORE_CI_TEXT(K2:K596)` |
| L602 | `=MEAN_SCORE_CI_TEXT(L2:L596)` |
| M602 | `=MEAN_SCORE_CI_TEXT(M2:M596)` |
| N602 | `=MEAN_SCORE_CI_TEXT(N2:N596)` |
| O602 | `=MEAN_SCORE_CI_TEXT(O2:O596)` |
| P602 | `=MEAN_SCORE_CI_TEXT(P2:P596)` |
| Q602 | `=MEAN_SCORE_CI_TEXT(Q2:Q596)` |
| R602 | `=MEAN_SCORE_CI_TEXT(R2:R596)` |
| G603 | `=MAE_CI_TEXT($F$2:$F$596, G2:G596) ` |
| H603 | `=MAE_CI_TEXT($F$2:$F$596, H2:H596) ` |
| I603 | `=MAE_CI_TEXT($F$2:$F$596, I2:I596) ` |
| J603 | `=MAE_CI_TEXT($F$2:$F$596, J2:J596) ` |
| K603 | `=MAE_CI_TEXT($F$2:$F$596, K2:K596) ` |
| L603 | `=MAE_CI_TEXT($F$2:$F$596, L2:L596) ` |
| M603 | `=MAE_CI_TEXT($F$2:$F$596, M2:M596) ` |
| N603 | `=MAE_CI_TEXT($F$2:$F$596, N2:N596) ` |
| O603 | `=MAE_CI_TEXT($F$2:$F$596, O2:O596) ` |
| P603 | `=MAE_CI_TEXT($F$2:$F$596, P2:P596) ` |
| Q603 | `=MAE_CI_TEXT($F$2:$F$596, Q2:Q596) ` |
| R603 | `=MAE_CI_TEXT($F$2:$F$596, R2:R596) ` |
| G605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, G2:G596)` |
| H605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, H2:H596)` |
| I605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, I2:I596)` |
| J605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, J2:J596)` |
| K605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, K2:K596)` |
| L605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, L2:L596)` |
| M605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, M2:M596)` |
| N605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, N2:N596)` |
| O605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, O2:O596)` |
| P605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, P2:P596)` |
| Q605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, Q2:Q596)` |
| R605 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$596, R2:R596)` |

### NMED Diagnosis (47 formulas)

| Cell | Formula |
|------|--------|
| I732 | `=MAE(   INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),   I2:I721 )` |
| J732 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  J2:J721)` |
| K732 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  K2:K721)` |
| L732 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  L2:L721)` |
| M732 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  M2:M721)` |
| N732 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  N2:N721)` |
| O732 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  O2:O721)` |
| G736 | `=MAE_IQR_TEXT($F$2:$F$721, G2:G721) ` |
| I736 | `=MAE_IQR_TEXT($F$2:$F$721, I2:I721) ` |
| J736 | `=MAE_IQR_TEXT($F$2:$F$721, J2:J721) ` |
| K736 | `=MAE_IQR_TEXT($F$2:$F$721, K2:K721) ` |
| L736 | `=MAE_IQR_TEXT($F$2:$F$721, L2:L721) ` |
| M736 | `=MAE_IQR_TEXT($F$2:$F$721, M2:M721) ` |
| N736 | `=MAE_IQR_TEXT($F$2:$F$721, N2:N721) ` |
| O736 | `=MAE_IQR_TEXT($F$2:$F$721, O2:O721) ` |
| G737 | `=MEDAE_IQR_TEXT($F$2:$F$721, G2:G721)` |
| I737 | `=MEDAE_IQR_TEXT($F$2:$F$721, I2:I721)` |
| J737 | `=MEDAE_IQR_TEXT($F$2:$F$721, J2:J721)` |
| K737 | `=MEDAE_IQR_TEXT($F$2:$F$721, K2:K721)` |
| L737 | `=MEDAE_IQR_TEXT($F$2:$F$721, L2:L721)` |
| M737 | `=MEDAE_IQR_TEXT($F$2:$F$721, M2:M721)` |
| N737 | `=MEDAE_IQR_TEXT($F$2:$F$721, N2:N721)` |
| O737 | `=MEDAE_IQR_TEXT($F$2:$F$721, O2:O721)` |
| G738 | `=MEAN_SCORE_CI_TEXT(G2:G721)` |
| I738 | `=MEAN_SCORE_CI_TEXT(I2:I721)` |
| J738 | `=MEAN_SCORE_CI_TEXT(J2:J721)` |
| K738 | `=MEAN_SCORE_CI_TEXT(K2:K721)` |
| L738 | `=MEAN_SCORE_CI_TEXT(L2:L721)` |
| M738 | `=MEAN_SCORE_CI_TEXT(M2:M721)` |
| N738 | `=MEAN_SCORE_CI_TEXT(N2:N721)` |
| O738 | `=MEAN_SCORE_CI_TEXT(O2:O721)` |
| G739 | `=MAE_CI_TEXT($F$2:$F$721, G2:G721) ` |
| I739 | `=MAE_CI_TEXT($F$2:$F$721, I2:I721) ` |
| J739 | `=MAE_CI_TEXT($F$2:$F$721, J2:J721) ` |
| K739 | `=MAE_CI_TEXT($F$2:$F$721, K2:K721) ` |
| L739 | `=MAE_CI_TEXT($F$2:$F$721, L2:L721) ` |
| M739 | `=MAE_CI_TEXT($F$2:$F$721, M2:M721) ` |
| N739 | `=MAE_CI_TEXT($F$2:$F$721, N2:N721) ` |
| O739 | `=MAE_CI_TEXT($F$2:$F$721, O2:O721) ` |
| G741 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, G2:G721)` |
| I741 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, I2:I721)` |
| J741 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, J2:J721)` |
| K741 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, K2:K721)` |
| L741 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, L2:L721)` |
| M741 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, M2:M721)` |
| N741 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, N2:N721)` |
| O741 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, O2:O721)` |

### Copy of NMED Diagnosis (60 formulas)

| Cell | Formula |
|------|--------|
| G723 | `=MAE(   INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),   G2:G721 )` |
| H723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  H2:H721)` |
| I723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  I2:I721)` |
| J723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  J2:J721)` |
| K723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  K2:K721)` |
| L723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  L2:L721)` |
| M723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  M2:M721)` |
| N723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  N2:N721)` |
| O723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  O2:O721)` |
| P723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  P2:P721)` |
| Q723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  Q2:Q721)` |
| R723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  R2:R721)` |
| G727 | `=MAE_IQR_TEXT($F$2:$F$721, G2:G721) ` |
| H727 | `=MAE_IQR_TEXT($F$2:$F$721, H2:H721) ` |
| I727 | `=MAE_IQR_TEXT($F$2:$F$721, I2:I721) ` |
| J727 | `=MAE_IQR_TEXT($F$2:$F$721, J2:J721) ` |
| K727 | `=MAE_IQR_TEXT($F$2:$F$721, K2:K721) ` |
| L727 | `=MAE_IQR_TEXT($F$2:$F$721, L2:L721) ` |
| M727 | `=MAE_IQR_TEXT($F$2:$F$721, M2:M721) ` |
| N727 | `=MAE_IQR_TEXT($F$2:$F$721, N2:N721) ` |
| O727 | `=MAE_IQR_TEXT($F$2:$F$721, O2:O721) ` |
| P727 | `=MAE_IQR_TEXT($F$2:$F$721, P2:P721) ` |
| Q727 | `=MAE_IQR_TEXT($F$2:$F$721, Q2:Q721) ` |
| R727 | `=MAE_IQR_TEXT($F$2:$F$721, R2:R721) ` |
| G728 | `=MEDAE_IQR_TEXT($F$2:$F$721, G2:G721)` |
| H728 | `=MEDAE_IQR_TEXT($F$2:$F$721, H2:H721)` |
| I728 | `=MEDAE_IQR_TEXT($F$2:$F$721, I2:I721)` |
| J728 | `=MEDAE_IQR_TEXT($F$2:$F$721, J2:J721)` |
| K728 | `=MEDAE_IQR_TEXT($F$2:$F$721, K2:K721)` |
| L728 | `=MEDAE_IQR_TEXT($F$2:$F$721, L2:L721)` |
| M728 | `=MEDAE_IQR_TEXT($F$2:$F$721, M2:M721)` |
| N728 | `=MEDAE_IQR_TEXT($F$2:$F$721, N2:N721)` |
| O728 | `=MEDAE_IQR_TEXT($F$2:$F$721, O2:O721)` |
| P728 | `=MEDAE_IQR_TEXT($F$2:$F$721, P2:P721)` |
| Q728 | `=MEDAE_IQR_TEXT($F$2:$F$721, Q2:Q721)` |
| R728 | `=MEDAE_IQR_TEXT($F$2:$F$721, R2:R721)` |
| G729 | `=MEAN_SCORE_CI_TEXT(G2:G721)` |
| H729 | `=MEAN_SCORE_CI_TEXT(H2:H721)` |
| I729 | `=MEAN_SCORE_CI_TEXT(I2:I721)` |
| J729 | `=MEAN_SCORE_CI_TEXT(J2:J721)` |
| K729 | `=MEAN_SCORE_CI_TEXT(K2:K721)` |
| L729 | `=MEAN_SCORE_CI_TEXT(L2:L721)` |
| M729 | `=MEAN_SCORE_CI_TEXT(M2:M721)` |
| N729 | `=MEAN_SCORE_CI_TEXT(N2:N721)` |
| O729 | `=MEAN_SCORE_CI_TEXT(O2:O721)` |
| P729 | `=MEAN_SCORE_CI_TEXT(P2:P721)` |
| Q729 | `=MEAN_SCORE_CI_TEXT(Q2:Q721)` |
| R729 | `=MEAN_SCORE_CI_TEXT(R2:R721)` |
| G732 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, G2:G721, "greater")` |
| H732 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, H2:H721, "greater")` |
| I732 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, I2:I721, "greater")` |
| J732 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, J2:J721, "greater")` |
| K732 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, K2:K721, "greater")` |
| L732 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, L2:L721, "greater")` |
| M732 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, M2:M721, "greater")` |
| N732 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, N2:N721, "greater")` |
| O732 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, O2:O721, "greater")` |
| P732 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, P2:P721, "greater")` |
| Q732 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, Q2:Q721, "greater")` |
| R732 | `=WILCOXON_SIGNED_RANK_TEXT($F$2:$F$721, R2:R721, "greater")` |

### finetune-Eurorad (15 formulas)

| Cell | Formula |
|------|--------|
| G209 | `=MODEL_ACCURACY(   INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),   G2:G208 ) ` |
| H209 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  H2:H208) ` |
| I209 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  I2:I208) ` |
| J209 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  J2:J208) ` |
| K209 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  K2:K208) ` |
| L209 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  L2:L208) ` |
| M209 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  M2:M208) ` |
| N209 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  N2:N208) ` |
| O209 | `=MODEL_ACCURACY( INDEX(2:208, , MATCH("FinalDiagnosis", 1:1, 0)),  O2:O208) ` |
| H211 | `=AVERAGE(G209:I209)` |
| K211 | `=AVERAGE(J209:L209)` |
| N211 | `=AVERAGE(M209:O209)` |
| Q211 | `=AVERAGE(P209:R209)` |
| N214 | `=SC_ACCURACY_WILSON($F$2:$F$208,M2:M208,N2:N208,O2:O208)` |
| M216 | `=AVERAGE(M209:O209)` |

### NMED Diagnosis beams (13 formulas)

| Cell | Formula |
|------|--------|
| G723 | `=MAE(   INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),   G2:G721 )` |
| H723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  H2:H721)` |
| I723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  I2:I721)` |
| J723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  J2:J721)` |
| K723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  K2:K721)` |
| L723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  L2:L721)` |
| M723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  M2:M721)` |
| N723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  N2:N721)` |
| O723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  O2:O721)` |
| P723 | `=MAE( INDEX(2:721, , MATCH("HumanEvalScore", 1:1, 0)),  P2:P721)` |
| H725 | `=LIKERT_CONSENSUS_MAE_CI(Table5[HumanEvalScore], Table5[oss20b-5beams-v1],Table5[oss20b-5beams-v2], Table5[oss20b-5beams-v3])` |
| K725 | `=LIKERT_CONSENSUS_MAE_CI(Table5[HumanEvalScore],Table5[oss20b-9beams-v1],Table5[oss20b-9beams-v2],Table5[oss20b-9beams-v3])` |
| N725 | `=LIKERT_CONSENSUS_MAE_CI(Table5[HumanEvalScore],Table5[oss20b-11beams-v1],Table5[oss20b-11beams-v2],Table5[oss20b-11beams-v3])` |

### NMED Treatment beams (13 formulas)

| Cell | Formula |
|------|--------|
| G598 | `=MAE(   INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),   G2:G596 )` |
| H598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  H2:H596)` |
| I598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  I2:I596)` |
| J598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  J2:J596)` |
| K598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  K2:K596)` |
| L598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  L2:L596)` |
| M598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  M2:M596)` |
| N598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  N2:N596)` |
| O598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  O2:O596)` |
| P598 | `=MAE( INDEX(2:596, , MATCH("HumanEvalScore", 1:1, 0)),  P2:P596)` |
| H601 | `=LIKERT_CONSENSUS_MAE_CI(Table4[HumanEvalScore],Table4[5beams-v1],Table4[5beams-v2], Table4[5beams-v3])` |
| K601 | `=LIKERT_CONSENSUS_MAE_CI(Table4[HumanEvalScore],Table4[9beams-v1],Table4[9beams-v2], Table4[9beams-v3])` |
| N601 | `=LIKERT_CONSENSUS_MAE_CI(Table4[HumanEvalScore],Table4[11beams-v1],Table4[11beams-v2], Table4[11beams-v3])` |

### oss20b_med_promptim_v1 (209 formulas)

| Cell | Formula |
|------|--------|
| H2 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G2,""<answer>(.*?)</answer>"") "),"Nail–patella syndrome")` |
| H3 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G3,""<answer>(.*?)</answer>"") "),"Angiosarcoma")` |
| H4 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G4,""<answer>(.*?)</answer>"") "),"Pylephlebitis and pancreatic parenchymal varices secondary to hepatic abscess")` |
| H5 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G5,""<answer>(.*?)</answer>"") "),"Pure yolk sac tumour")` |
| H6 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G6,""<answer>(.*?)</answer>"") "),"Atherosclerotic ICA stenosis with secondary ischemic events")` |
| H7 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G7,""<answer>(.*?)</answer>"") "),"Tumefactive demyelinating lesion")` |
| H8 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G8,""<answer>(.*?)</answer>"") "),"Cavitating mesenteric lymph node syndrome related to celiac disease")` |
| H9 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G9,""<answer>(.*?)</answer>"") "),"Ischiorectal liposarcoma")` |
| H10 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G10,""<answer>(.*?)</answer>"") "),"Hepatic hydatid disease (type IIb)")` |
| H11 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G11,""<answer>(.*?)</answer>"") "),"Spondylothoracic dysostosis (Lavy–Moseley syndrome)")` |
| H12 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G12,""<answer>(.*?)</answer>"") "),"Rhomboid fossa variant")` |
| H13 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G13,""<answer>(.*?)</answer>"") "),"Self-limiting sternal tumour of childhood (SELSTOC)")` |
| H14 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G14,""<answer>(.*?)</answer>"") "),"OEIS (omphalocele")` |
| H15 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G15,""<answer>(.*?)</answer>"") "),"Acromioclavicular cyst secondary to massive rotator cuff tear")` |
| H16 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G16,""<answer>(.*?)</answer>"") "),"Dirofilariasis")` |
| H17 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G17,""<answer>(.*?)</answer>"") "),"Gardner’s syndrome")` |
| H18 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G18,""<answer>(.*?)</answer>"") "),"Atypically located third or fourth infected branchial cleft cyst")` |
| H19 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G19,""<answer>(.*?)</answer>"") "),"Acute postoperative parotitis")` |
| H20 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G20,""<answer>(.*?)</answer>"") "),"Infantile myofibromatosis")` |
| H21 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G21,""<answer>(.*?)</answer>"") "),"Metastatic breast and axillary lesions from retroperitoneal sarcoma")` |
| H22 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G22,""<answer>(.*?)</answer>"") "),"Humeral avulsion of glenohumeral ligament (HALG infection) with superior labrum from anterior to posterior (SLAP) lesio...` |
| H23 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G23,""<answer>(.*?)</answer>"") "),"Pulmonary arteriovenous malformations")` |
| H24 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G24,""<answer>(.*?)</answer>"") "),"Neurogenic heterotopic ossification")` |
| H25 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G25,""<answer>(.*?)</answer>"") "),"Calcium pyrophosphate deposition disease")` |
| H26 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G26,""<answer>(.*?)</answer>"") "),"Malignant mitral valve prolapse with subendocardial late enhancement")` |
| H27 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G27,""<answer>(.*?)</answer>"") "),"Fibrous hamartoma of infancy")` |
| H28 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G28,""<answer>(.*?)</answer>"") "),"Post-transplant ureteral stent removal")` |
| H29 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G29,""<answer>(.*?)</answer>"") "),"Pelvic paraganglioma")` |
| H30 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G30,""<answer>(.*?)</answer>"") "),"White cord syndrome")` |
| H31 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G31,""<answer>(.*?)</answer>"") "),"Subacute combined degeneration secondary to abuse of nitrous oxide")` |
| H32 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G32,""<answer>(.*?)</answer>"") "),"Diffuse neurofibromatosis tissue in neurofibromatosis type 1")` |
| H33 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G33,""<answer>(.*?)</answer>"") "),"Mondor’s disease")` |
| H34 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G34,""<answer>(.*?)</answer>"") "),"Fibrous dysplasia")` |
| H35 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G35,""<answer>(.*?)</answer>"") "),"Calcinosis (in the setting of limited cutaneous systemic sclerosis)")` |
| H36 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G36,""<answer>(.*?)</answer>"") "),"Boerhaave syndrome")` |
| H37 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G37,""<answer>(.*?)</answer>"") "),"Achilles rupture with tennis leg")` |
| H38 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G38,""<answer>(.*?)</answer>"") "),"Immune-related pancreatitis")` |
| H39 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G39,""<answer>(.*?)</answer>"") "),"Extra-pleural haematoma")` |
| H40 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G40,""<answer>(.*?)</answer>"") "),"Rasmussen aneurysm")` |
| H41 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G41,""<answer>(.*?)</answer>"") "),"Lymphoma")` |
| H42 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G42,""<answer>(.*?)</answer>"") "),"Intercostal nerve schwannoma (neurilemoma)")` |
| H43 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G43,""<answer>(.*?)</answer>"") "),"Splenogonadal fusion")` |
| H44 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G44,""<answer>(.*?)</answer>"") "),"Duodenum inversum")` |
| H45 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G45,""<answer>(.*?)</answer>"") "),"Pituitary macroadenoma")` |
| H46 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G46,""<answer>(.*?)</answer>"") "),"Subacute cerebral infarcts")` |
| H47 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G47,""<answer>(.*?)</answer>"") "),"Choroidal detachment and posterior scleritis")` |
| H48 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G48,""<answer>(.*?)</answer>"") "),"Primary hyperphosphatemic tumoral calcinosis")` |
| H49 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G49,""<answer>(.*?)</answer>"") "),"Paroxysmal nocturnal haemoglobinuria")` |
| H50 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G50,""<answer>(.*?)</answer>"") "),"SMART syndrome")` |
| H51 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G51,""<answer>(.*?)</answer>"") "),"Liposarcoma")` |
| H52 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G52,""<answer>(.*?)</answer>"") "),"Hydrometrocolpos")` |
| H53 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G53,""<answer>(.*?)</answer>"") "),"Creutzfeldt–Jakob disease")` |
| H54 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G54,""<answer>(.*?)</answer>"") "),"Hidradenitis suppurativa")` |
| H55 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G55,""<answer>(.*?)</answer>"") "),"Hepatic subcapsular cerebrospinal fluid pseudocyst")` |
| H56 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G56,""<answer>(.*?)</answer>"") "),"Morel-Lavallée lesion in a patient with previous myxoid liposarcoma")` |
| H57 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G57,""<answer>(.*?)</answer>"") "),"Calcified cavernoma")` |
| H58 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G58,""<answer>(.*?)</answer>"") "),"Synchronous squamous cell carcinoma and neuroendocrine tumour")` |
| H59 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G59,""<answer>(.*?)</answer>"") "),"Cecal volvulus")` |
| H60 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G60,""<answer>(.*?)</answer>"") "),"Critical illness-associate cerebral microbleeds")` |
| H61 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G61,""<answer>(.*?)</answer>"") "),"Primary intravascular synovial sarcoma")` |
| H62 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G62,""<answer>(.*?)</answer>"") "),"Ketamine bladder")` |
| H63 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G63,""<answer>(.*?)</answer>"") "),"Mucinous cystic neoplasm")` |
| H64 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G64,""<answer>(.*?)</answer>"") "),"Friedreich’s ataxia")` |
| H65 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G65,""<answer>(.*?)</answer>"") "),"Pleural splenosis")` |
| H66 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G66,""<answer>(.*?)</answer>"") "),"Inflammatory cerebral amyloid angiopathy")` |
| H67 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G67,""<answer>(.*?)</answer>"") "),"Anastomotic pseudoaneurysm (PTFE brachiocephalic bypass graft)")` |
| H68 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G68,""<answer>(.*?)</answer>"") "),"Fibroadenoma")` |
| H69 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G69,""<answer>(.*?)</answer>"") "),"Von Hippel–Lindau syndrome (VHL)")` |
| H70 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G70,""<answer>(.*?)</answer>"") "),"Skull base chondrosarcoma")` |
| H71 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G71,""<answer>(.*?)</answer>"") "),"Bone marrow oedema syndrome")` |
| H72 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G72,""<answer>(.*?)</answer>"") "),"Homocystinuria")` |
| H73 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G73,""<answer>(.*?)</answer>"") "),"Acute vasitis")` |
| H74 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G74,""<answer>(.*?)</answer>"") "),"Infective aetiology (tuberculous/fungal)")` |
| H75 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G75,""<answer>(.*?)</answer>"") "),"Primary hyperparathyroidism with fibrous dysplasia of bone")` |
| H76 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G76,""<answer>(.*?)</answer>"") "),"Atraumatic compartment syndrome")` |
| H77 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G77,""<answer>(.*?)</answer>"") "),"Simultaneous lacrimal and parotid gland lymphoma")` |
| H78 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G78,""<answer>(.*?)</answer>"") "),"Herpes simplex encephalitis")` |
| H79 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G79,""<answer>(.*?)</answer>"") "),"Neurocysticercosis with disseminated cysticercosis")` |
| H80 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G80,""<answer>(.*?)</answer>"") "),"Hibernoma")` |
| H81 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G81,""<answer>(.*?)</answer>"") "),"Metastatic melanoma in the gallbladder")` |
| H82 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G82,""<answer>(.*?)</answer>"") "),"Cochlear incomplete partition type II (IP-2)")` |
| H83 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G83,""<answer>(.*?)</answer>"") "),"Biventricular endomyocardial fibrosis [4")` |
| H84 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G84,""<answer>(.*?)</answer>"") "),"Liver pyogenic abscess")` |
| H85 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G85,""<answer>(.*?)</answer>"") "),"Telangiectatic osteosarcoma")` |
| H86 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G86,""<answer>(.*?)</answer>"") "),"Fahr’s disease")` |
| H87 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G87,""<answer>(.*?)</answer>"") "),"Polyostotic melorheostosis")` |
| H88 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G88,""<answer>(.*?)</answer>"") "),"Isolated right tubal torsion with haematosalpinx")` |
| H89 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G89,""<answer>(.*?)</answer>"") "),"Low phospholipid-associated cholelithiasis (LPAC) syndrome")` |
| H90 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G90,""<answer>(.*?)</answer>"") "),"WHO grade 2 bilateral frontal oligodendroglioma")` |
| H91 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G91,""<answer>(.*?)</answer>"") "),"Oesophageal lymphoma")` |
| H92 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G92,""<answer>(.*?)</answer>"") "),"Osteochondromas of the ventral and dorsal scapula")` |
| H93 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G93,""<answer>(.*?)</answer>"") "),"Lymphoma")` |
| H94 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G94,""<answer>(.*?)</answer>"") "),"Low-grade chondrosarcoma")` |
| H95 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G95,""<answer>(.*?)</answer>"") "),"Desmoid fibromatosis")` |
| H96 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G96,""<answer>(.*?)</answer>"") "),"Lesser omental infarction")` |
| H97 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G97,""<answer>(.*?)</answer>"") "),"Accessory liver lobe")` |
| H98 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G98,""<answer>(.*?)</answer>"") "),"Meconium periorchitis")` |
| H99 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G99,""<answer>(.*?)</answer>"") "),"Parosteal lipoma")` |
| H100 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G100,""<answer>(.*?)</answer>"") "),"Usual interstitial pneumonia due to idiopathic pulmonary fibrosis")` |
| H101 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G101,""<answer>(.*?)</answer>"") "),"Epipericardial fat necrosis")` |
| H102 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G102,""<answer>(.*?)</answer>"") "),"Overshunting-associated myelopathy")` |
| H103 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G103,""<answer>(.*?)</answer>"") "),"Benign metastasising leiomyoma")` |
| H104 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G104,""<answer>(.*?)</answer>"") "),"Mixed sclerosing bone dysplasia")` |
| H105 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G105,""<answer>(.*?)</answer>"") "),"Trochlear dysplasia (Dejour type B)")` |
| H106 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G106,""<answer>(.*?)</answer>"") "),"Complete hydatidiform mole")` |
| H107 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G107,""<answer>(.*?)</answer>"") "),"Hepatic cholangiocarcinoma")` |
| H108 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G108,""<answer>(.*?)</answer>"") "),"Tailgut cyst")` |
| H109 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G109,""<answer>(.*?)</answer>"") "),"Right paravesical hernia")` |
| H110 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G110,""<answer>(.*?)</answer>"") "),"Oesophageal mucocele post corrosive stricture surgery")` |
| H111 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G111,""<answer>(.*?)</answer>"") "),"Peritoneal hydatidosis")` |
| H112 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G112,""<answer>(.*?)</answer>"") "),"Invasive fungal rhinosinusitis")` |
| H113 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G113,""<answer>(.*?)</answer>"") "),"Other aggressive lymphoma")` |
| H114 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G114,""<answer>(.*?)</answer>"") "),"Ganglioneuroma")` |
| H115 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G115,""<answer>(.*?)</answer>"") "),"Slipping rib syndrome")` |
| H116 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G116,""<answer>(.*?)</answer>"") "),"Peritoneal carcinomatosis")` |
| H117 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G117,""<answer>(.*?)</answer>"") "),"Moyamoya disease")` |
| H118 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G118,""<answer>(.*?)</answer>"") "),"Traumatic brachial plexopathy")` |
| H119 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G119,""<answer>(.*?)</answer>"") "),"Lipoma of the tibialis anterior tendon sheath")` |
| H120 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G120,""<answer>(.*?)</answer>"") "),"Wilson’s disease")` |
| H121 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G121,""<answer>(.*?)</answer>"") "),"Malignant neoplasia")` |
| H122 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G122,""<answer>(.*?)</answer>"") "),"Breast cancer metastasis")` |
| H123 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G123,""<answer>(.*?)</answer>"") "),"OHVIRA / Herlyn–Werner–Wunderlich syndrome")` |
| H124 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G124,""<answer>(.*?)</answer>"") "),"Unicentric Castleman disease")` |
| H125 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G125,""<answer>(.*?)</answer>"") "),"Rectal gastrointestinal stromal tumour")` |
| H126 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G126,""<answer>(.*?)</answer>"") "),"Vesico-intestinal fistula")` |
| H127 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G127,""<answer>(.*?)</answer>"") "),"Usual interstitial pneumonia due to idiopathic pulmonary fibrosis")` |
| H128 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G128,""<answer>(.*?)</answer>"") "),"Spontaneous pneumothorax secondary to rheumatoid nodule")` |
| H129 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G129,""<answer>(.*?)</answer>"") "),"Linitis plastica in diffuse stomach carcinoma")` |
| H130 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G130,""<answer>(.*?)</answer>"") "),"Hepatocellular carcinoma")` |
| H131 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G131,""<answer>(.*?)</answer>"") "),"Acute Marchiafava–Bignami disease")` |
| H132 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G132,""<answer>(.*?)</answer>"") "),"Retropharyngeal bronchogenic cyst")` |
| H133 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G133,""<answer>(.*?)</answer>"") "),"Extraskeletal osteosarcoma")` |
| H134 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G134,""<answer>(.*?)</answer>"") "),"Cochlear hypoplasia type I (with aplasia of the cochlear nerve)")` |
| H135 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G135,""<answer>(.*?)</answer>"") "),"Xanthogranulomatous pyelonephritis")` |
| H136 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G136,""<answer>(.*?)</answer>"") "),"#N/A")` |
| H137 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G137,""<answer>(.*?)</answer>"") "),"Infantile cortical hyperostosis (Caffey disease)")` |
| H138 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G138,""<answer>(.*?)</answer>"") "),"Roberts syndrome")` |
| H139 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G139,""<answer>(.*?)</answer>"") "),"Chorea-acanthocytosis")` |
| H140 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G140,""<answer>(.*?)</answer>"") "),"Haemangioma")` |
| H141 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G141,""<answer>(.*?)</answer>"") "),"Lymphoma")` |
| H142 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G142,""<answer>(.*?)</answer>"") "),"Plasma cell mastitis")` |
| H143 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G143,""<answer>(.*?)</answer>"") "),"Leukoencephalopathy with brainstem and spinal cord involvement and lactate elevation (LBSL)")` |
| H144 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G144,""<answer>(.*?)</answer>"") "),"Hirayama disease")` |
| H145 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G145,""<answer>(.*?)</answer>"") "),"Intraductal papillary neoplasm of the bile duct")` |
| H146 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G146,""<answer>(.*?)</answer>"") "),"Autosplenectomy (rock spleen) secondary to sickle cell disease")` |
| H147 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G147,""<answer>(.*?)</answer>"") "),"Langer’s axillary arch")` |
| H148 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G148,""<answer>(.*?)</answer>"") "),"Müller–Weiss syndrome")` |
| H149 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G149,""<answer>(.*?)</answer>"") "),"Fibroadenoma")` |
| H150 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G150,""<answer>(.*?)</answer>"") "),"Paradoxical tuberculomas post-antituberculosis treatment")` |
| H151 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G151,""<answer>(.*?)</answer>"") "),"Caroli syndrome (congenital hepatic fibrosis with Caroli disease)")` |
| H152 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G152,""<answer>(.*?)</answer>"") "),"Pantothenate kinase-associated neurodegeneration")` |
| H153 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G153,""<answer>(.*?)</answer>"") "),"Langerhans cell histiocytosis")` |
| H154 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G154,""<answer>(.*?)</answer>"") "),"Giant microcystic serous cystadenoma of the pancreas")` |
| H155 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G155,""<answer>(.*?)</answer>"") "),"Malignant peripheral nerve sheath tumour")` |
| H156 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G156,""<answer>(.*?)</answer>"") "),"Disseminated AIDS-related Kaposi sarcoma")` |
| H157 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G157,""<answer>(.*?)</answer>"") "),"Neonatal mastitis with abscess")` |
| H158 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G158,""<answer>(.*?)</answer>"") "),"Fungal osteomyelitis of the patella")` |
| H159 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G159,""<answer>(.*?)</answer>"") "),"Cervical agenesis with endometrioma")` |
| H160 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G160,""<answer>(.*?)</answer>"") "),"Neurofibromas")` |
| H161 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G161,""<answer>(.*?)</answer>"") "),"Uremic encephalopathy")` |
| H162 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G162,""<answer>(.*?)</answer>"") "),"Dancing megasperm")` |
| H163 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G163,""<answer>(.*?)</answer>"") "),"Lymphoma")` |
| H164 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G164,""<answer>(.*?)</answer>"") "),"Infantile cortical hyperostosis (Caffey disease)")` |
| H165 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G165,""<answer>(.*?)</answer>"") "),"Split hand/split foot malformation (SHFM)")` |
| H166 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G166,""<answer>(.*?)</answer>"") "),"Spinal epidural abscess")` |
| H167 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G167,""<answer>(.*?)</answer>"") "),"Neurocutaneous melanosis")` |
| H168 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G168,""<answer>(.*?)</answer>"") "),"Cystic fibrosis")` |
| H169 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G169,""<answer>(.*?)</answer>"") "),"Osteoblastoma")` |
| H170 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G170,""<answer>(.*?)</answer>"") "),"Plexiform neurofibroma")` |
| H171 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G171,""<answer>(.*?)</answer>"") "),"Mycetoma foot")` |
| H172 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G172,""<answer>(.*?)</answer>"") "),"Aquaporin-4 IgG-positive neuromyelitis optica spectrum disorders (AQP4-NMOSD)")` |
| H173 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G173,""<answer>(.*?)</answer>"") "),"Retropancreatic mature teratoma")` |
| H174 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G174,""<answer>(.*?)</answer>"") "),"Multicystic lymphangioma")` |
| H175 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G175,""<answer>(.*?)</answer>"") "),"Rosette-forming glioneuronal tumour (RGNT)")` |
| H176 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G176,""<answer>(.*?)</answer>"") "),"Hibernoma")` |
| H177 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G177,""<answer>(.*?)</answer>"") "),"Dysgerminoma")` |
| H178 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G178,""<answer>(.*?)</answer>"") "),"Haematometrocolpos")` |
| H179 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G179,""<answer>(.*?)</answer>"") "),"Ewing sarcoma (or other small round blue cell tumour)")` |
| H180 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G180,""<answer>(.*?)</answer>"") "),"Spinal neurosarcoidosis with trident sign")` |
| H181 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G181,""<answer>(.*?)</answer>"") "),"Pyogenic spondylodiscitis")` |
| H182 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G182,""<answer>(.*?)</answer>"") "),"Hereditary haemorrhagic telangiectasia")` |
| H183 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G183,""<answer>(.*?)</answer>"") "),"Haemangioblastoma")` |
| H184 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G184,""<answer>(.*?)</answer>"") "),"First branchial cleft cyst")` |
| H185 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G185,""<answer>(.*?)</answer>"") "),"Acromegaly secondary to pituitary macroadenoma")` |
| H186 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G186,""<answer>(.*?)</answer>"") "),"Wallerian degeneration of pontocerebellar tracts secondary to unilateral pontine infarction")` |
| H187 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G187,""<answer>(.*?)</answer>"") "),"Stump appendicitis")` |
| H188 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G188,""<answer>(.*?)</answer>"") "),"Xanthogranulomatous pyelonephritis (diffuse form)")` |
| H189 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G189,""<answer>(.*?)</answer>"") "),"Adrenal adenocarcinoma")` |
| H190 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G190,""<answer>(.*?)</answer>"") "),"Classical osteosarcoma")` |
| H191 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G191,""<answer>(.*?)</answer>"") "),"Primary prostatic urethral neoplasm")` |
| H192 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G192,""<answer>(.*?)</answer>"") "),"Infection by Mycobacterium tuberculosis")` |
| H193 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G193,""<answer>(.*?)</answer>"") "),"Anaplastic meningioma (WHO Grade III)")` |
| H194 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G194,""<answer>(.*?)</answer>"") "),"Pott’s spine with psoas abscess")` |
| H195 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G195,""<answer>(.*?)</answer>"") "),"Intramuscular haemangioma")` |
| H196 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G196,""<answer>(.*?)</answer>"") "),"Congenital femoral deficiency and absence of the anterior cruciate ligament")` |
| H197 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G197,""<answer>(.*?)</answer>"") "),"Milwaukee shoulder syndrome")` |
| H198 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G198,""<answer>(.*?)</answer>"") "),"Rhombencephalosynapsis")` |
| H199 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G199,""<answer>(.*?)</answer>"") "),"Pyogenic liver abscess")` |
| H200 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G200,""<answer>(.*?)</answer>"") "),"Progressive fibrodysplasia ossificans")` |
| H201 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G201,""<answer>(.*?)</answer>"") "),"Multiple hereditary exostoses")` |
| H202 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G202,""<answer>(.*?)</answer>"") "),"Sinding–Larsen–Johansson disease (SLJD)")` |
| H203 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G203,""<answer>(.*?)</answer>"") "),"Haberland syndrome (Encephalocraniocutaneous lipomatosis)")` |
| H204 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G204,""<answer>(.*?)</answer>"") "),"Pilocytic astrocytoma")` |
| H205 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G205,""<answer>(.*?)</answer>"") "),"Probable Creutzfeldt–Jakob disease")` |
| H206 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G206,""<answer>(.*?)</answer>"") "),"Lipoidal proteinosis")` |
| H207 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G207,""<answer>(.*?)</answer>"") "),"Rheumatoid arthritis")` |
| H208 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G208,""<answer>(.*?)</answer>"") "),"Giant caseating cerebellar tuberculoma")` |
| H209 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G209,""<answer>(.*?)</answer>"") "),"Sturge–Weber syndrome")` |
| H210 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G210,""<answer>(.*?)</answer>"") "),"Cochlear nerve deficiency")` |

### oss20b_high_promptim_v1 (209 formulas)

| Cell | Formula |
|------|--------|
| H2 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G2,""<answer>(.*?)</answer>"") "),"Nail–patella syndrome")` |
| H3 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G3,""<answer>(.*?)</answer>"") "),"Angiosarcoma")` |
| H4 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G4,""<answer>(.*?)</answer>"") "),"Pylephlebitis and pancreatic parenchymal varices secondary to hepatic abscess")` |
| H5 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G5,""<answer>(.*?)</answer>"") "),"Pure yolk sac tumour")` |
| H6 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G6,""<answer>(.*?)</answer>"") "),"Atherosclerotic ICA stenosis with secondary ischemic events")` |
| H7 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G7,""<answer>(.*?)</answer>"") "),"Tumefactive demyelinating lesion")` |
| H8 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G8,""<answer>(.*?)</answer>"") "),"Whipple’s disease")` |
| H9 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G9,""<answer>(.*?)</answer>"") "),"Ischiorectal liposarcoma")` |
| H10 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G10,""<answer>(.*?)</answer>"") "),"Hepatic hydatid disease (type IIb)")` |
| H11 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G11,""<answer>(.*?)</answer>"") "),"Spondylocostal dysostosis (Jarcho–Levin syndrome)")` |
| H12 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G12,""<answer>(.*?)</answer>"") "),"Rhomboid fossa variant")` |
| H13 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G13,""<answer>(.*?)</answer>"") "),"Self-limiting sternal tumour of childhood (SELSTOC)")` |
| H14 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G14,""<answer>(.*?)</answer>"") "),"OEIS (omphalocele, cloacal exstrophy, imperforate anus, and spinal dysraphism) complex")` |
| H15 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G15,""<answer>(.*?)</answer>"") "),"Acromioclavicular cyst secondary to massive rotator cuff tear")` |
| H16 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G16,""<answer>(.*?)</answer>"") "),"Dirofilariasis")` |
| H17 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G17,""<answer>(.*?)</answer>"") "),"Gardner’s syndrome")` |
| H18 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G18,""<answer>(.*?)</answer>"") "),"Atypically located third or fourth infected branchial cleft cyst")` |
| H19 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G19,""<answer>(.*?)</answer>"") "),"Acute postoperative parotitis")` |
| H20 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G20,""<answer>(.*?)</answer>"") "),"Infantile myofibromatosis")` |
| H21 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G21,""<answer>(.*?)</answer>"") "),"Metastatic breast and axillary lesions from retroperitoneal sarcoma")` |
| H22 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G22,""<answer>(.*?)</answer>"") "),"Humeral avulsion of glenohumeral ligament (HALG injury) with superior labrum from anterior to posterior (SLAP) lesion")` |
| H23 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G23,""<answer>(.*?)</answer>"") "),"Pulmonary arteriovenous malformations")` |
| H24 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G24,""<answer>(.*?)</answer>"") "),"Neurogenic heterotopic ossification")` |
| H25 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G25,""<answer>(.*?)</answer>"") "),"Hydroxyapatite deposits")` |
| H26 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G26,""<answer>(.*?)</answer>"") "),"Malignant mitral valve prolapse with subendocardial late enhancement")` |
| H27 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G27,""<answer>(.*?)</answer>"") "),"Fibrous hamartoma of infancy")` |
| H28 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G28,""<answer>(.*?)</answer>"") "),"Post-transplant ureteral stent removal")` |
| H29 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G29,""<answer>(.*?)</answer>"") "),"Pelvic paraganglioma")` |
| H30 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G30,""<answer>(.*?)</answer>"") "),"White cord syndrome")` |
| H31 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G31,""<answer>(.*?)</answer>"") "),"Subacute combined degeneration secondary to abuse of nitrous oxide")` |
| H32 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G32,""<answer>(.*?)</answer>"") "),"Diffuse neurofibromatosis tissue in neurofibromatosis type 1")` |
| H33 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G33,""<answer>(.*?)</answer>"") "),"Mondor’s disease")` |
| H34 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G34,""<answer>(.*?)</answer>"") "),"Brown tumour")` |
| H35 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G35,""<answer>(.*?)</answer>"") "),"Calcinosis (in the setting of limited cutaneous systemic sclerosis)")` |
| H36 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G36,""<answer>(.*?)</answer>"") "),"Boerhaave syndrome")` |
| H37 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G37,""<answer>(.*?)</answer>"") "),"Achilles rupture with tennis leg")` |
| H38 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G38,""<answer>(.*?)</answer>"") "),"Immune-related pancreatitis")` |
| H39 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G39,""<answer>(.*?)</answer>"") "),"Extra-pleural haematoma")` |
| H40 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G40,""<answer>(.*?)</answer>"") "),"Rasmussen aneurysm")` |
| H41 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G41,""<answer>(.*?)</answer>"") "),"#N/A")` |
| H42 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G42,""<answer>(.*?)</answer>"") "),"Intercostal nerve schwannoma (neurilemoma)")` |
| H43 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G43,""<answer>(.*?)</answer>"") "),"Splenogonadal fusion")` |
| H44 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G44,""<answer>(.*?)</answer>"") "),"Duodenum inversum")` |
| H45 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G45,""<answer>(.*?)</answer>"") "),"Pituitary macroadenoma")` |
| H46 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G46,""<answer>(.*?)</answer>"") "),"Non-ischemic cerebral enhancing (NICE) lesions")` |
| H47 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G47,""<answer>(.*?)</answer>"") "),"Choroidal detachment and posterior scleritis")` |
| H48 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G48,""<answer>(.*?)</answer>"") "),"Primary hyperphosphatemic tumoral calcinosis")` |
| H49 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G49,""<answer>(.*?)</answer>"") "),"Paroxysmal nocturnal haemoglobinuria")` |
| H50 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G50,""<answer>(.*?)</answer>"") "),"SMART syndrome")` |
| H51 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G51,""<answer>(.*?)</answer>"") "),"Liposarcoma")` |
| H52 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G52,""<answer>(.*?)</answer>"") "),"Pyometrocolpos")` |
| H53 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G53,""<answer>(.*?)</answer>"") "),"Creutzfeldt–Jakob disease")` |
| H54 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G54,""<answer>(.*?)</answer>"") "),"Hidradenitis suppurativa")` |
| H55 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G55,""<answer>(.*?)</answer>"") "),"Hepatic subcapsular cerebrospinal fluid pseudocyst")` |
| H56 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G56,""<answer>(.*?)</answer>"") "),"Morel-Lavallée lesion in a patient with previous myxoid liposarcoma")` |
| H57 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G57,""<answer>(.*?)</answer>"") "),"Calcified cavernoma")` |
| H58 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G58,""<answer>(.*?)</answer>"") "),"Synchronous squamous cell carcinoma and neuroendocrine tumour")` |
| H59 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G59,""<answer>(.*?)</answer>"") "),"Cecal volvulus")` |
| H60 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G60,""<answer>(.*?)</answer>"") "),"Critical illness-associate cerebral microbleeds")` |
| H61 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G61,""<answer>(.*?)</answer>"") "),"Primary intravascular synovial sarcoma")` |
| H62 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G62,""<answer>(.*?)</answer>"") "),"Ketamine bladder")` |
| H63 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G63,""<answer>(.*?)</answer>"") "),"Mucinous cystic neoplasm")` |
| H64 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G64,""<answer>(.*?)</answer>"") "),"Friedreich’s ataxia")` |
| H65 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G65,""<answer>(.*?)</answer>"") "),"Pleural splenosis")` |
| H66 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G66,""<answer>(.*?)</answer>"") "),"Inflammatory cerebral amyloid angiopathy")` |
| H67 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G67,""<answer>(.*?)</answer>"") "),"Anastomotic pseudoaneurysm (PTFE brachiocephalic bypass graft)")` |
| H68 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G68,""<answer>(.*?)</answer>"") "),"Fibroadenoma")` |
| H69 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G69,""<answer>(.*?)</answer>"") "),"Von Hippel–Lindau syndrome (VHL)")` |
| H70 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G70,""<answer>(.*?)</answer>"") "),"Skull base chordoma")` |
| H71 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G71,""<answer>(.*?)</answer>"") "),"Bone marrow oedema syndrome")` |
| H72 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G72,""<answer>(.*?)</answer>"") "),"Homocystinuria")` |
| H73 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G73,""<answer>(.*?)</answer>"") "),"Acute vasitis")` |
| H74 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G74,""<answer>(.*?)</answer>"") "),"Infective aetiology (tuberculous/fungal)")` |
| H75 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G75,""<answer>(.*?)</answer>"") "),"Primary hyperparathyroidism with fibrous dysplasia of bone")` |
| H76 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G76,""<answer>(.*?)</answer>"") "),"Atraumatic compartment syndrome")` |
| H77 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G77,""<answer>(.*?)</answer>"") "),"Simultaneous lacrimal and parotid gland lymphoma")` |
| H78 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G78,""<answer>(.*?)</answer>"") "),"Herpes simplex encephalitis")` |
| H79 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G79,""<answer>(.*?)</answer>"") "),"Neurocysticercosis with disseminated cysticercosis")` |
| H80 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G80,""<answer>(.*?)</answer>"") "),"Hibernoma")` |
| H81 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G81,""<answer>(.*?)</answer>"") "),"Metastatic melanoma in the gallbladder")` |
| H82 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G82,""<answer>(.*?)</answer>"") "),"Cochlear incomplete partition type III (IP-3)")` |
| H83 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G83,""<answer>(.*?)</answer>"") "),"Biventricular endomyocardial fibrosis [4")` |
| H84 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G84,""<answer>(.*?)</answer>"") "),"Liver pyogenic abscess")` |
| H85 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G85,""<answer>(.*?)</answer>"") "),"Telangiectatic osteosarcoma")` |
| H86 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G86,""<answer>(.*?)</answer>"") "),"Fahr’s disease")` |
| H87 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G87,""<answer>(.*?)</answer>"") "),"Polyostotic melorheostosis")` |
| H88 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G88,""<answer>(.*?)</answer>"") "),"Isolated right tubal torsion with haematosalpinx")` |
| H89 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G89,""<answer>(.*?)</answer>"") "),"Recurrent pyogenic cholangiohepatitis")` |
| H90 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G90,""<answer>(.*?)</answer>"") "),"WHO grade 2 bilateral frontal oligodendroglioma")` |
| H91 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G91,""<answer>(.*?)</answer>"") "),"Oesophageal lymphoma")` |
| H92 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G92,""<answer>(.*?)</answer>"") "),"Osteochondromas of the ventral and dorsal scapula")` |
| H93 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G93,""<answer>(.*?)</answer>"") "),"Solitary bone plasmacytoma")` |
| H94 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G94,""<answer>(.*?)</answer>"") "),"Enchondroma")` |
| H95 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G95,""<answer>(.*?)</answer>"") "),"Desmoid fibromatosis")` |
| H96 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G96,""<answer>(.*?)</answer>"") "),"Lesser omental infarction")` |
| H97 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G97,""<answer>(.*?)</answer>"") "),"Ectopic liver lobe")` |
| H98 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G98,""<answer>(.*?)</answer>"") "),"Meconium periorchitis")` |
| H99 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G99,""<answer>(.*?)</answer>"") "),"Parosteal lipoma")` |
| H100 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G100,""<answer>(.*?)</answer>"") "),"Usual interstitial pneumonia due to idiopathic pulmonary fibrosis")` |
| H101 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G101,""<answer>(.*?)</answer>"") "),"Epipericardial fat necrosis")` |
| H102 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G102,""<answer>(.*?)</answer>"") "),"Overshunting-associated myelopathy")` |
| H103 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G103,""<answer>(.*?)</answer>"") "),"Benign metastasising leiomyoma")` |
| H104 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G104,""<answer>(.*?)</answer>"") "),"Mixed sclerosing bone dysplasia")` |
| H105 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G105,""<answer>(.*?)</answer>"") "),"Trochlear dysplasia (Dejour type B)")` |
| H106 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G106,""<answer>(.*?)</answer>"") "),"Complete hydatidiform mole")` |
| H107 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G107,""<answer>(.*?)</answer>"") "),"Hepatic cholangiocarcinoma")` |
| H108 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G108,""<answer>(.*?)</answer>"") "),"Tailgut cyst")` |
| H109 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G109,""<answer>(.*?)</answer>"") "),"Right paravesical hernia")` |
| H110 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G110,""<answer>(.*?)</answer>"") "),"Oesophageal mucocele post corrosive stricture surgery")` |
| H111 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G111,""<answer>(.*?)</answer>"") "),"Peritoneal hydatidosis")` |
| H112 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G112,""<answer>(.*?)</answer>"") "),"Sphenoid sinus mycetoma")` |
| H113 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G113,""<answer>(.*?)</answer>"") "),"Metastatic disease")` |
| H114 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G114,""<answer>(.*?)</answer>"") "),"Ganglioneuroma")` |
| H115 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G115,""<answer>(.*?)</answer>"") "),"Slipping rib syndrome")` |
| H116 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G116,""<answer>(.*?)</answer>"") "),"Peritoneal carcinomatosis")` |
| H117 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G117,""<answer>(.*?)</answer>"") "),"Moyamoya disease")` |
| H118 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G118,""<answer>(.*?)</answer>"") "),"Traumatic brachial plexopathy")` |
| H119 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G119,""<answer>(.*?)</answer>"") "),"Lipoma of the tibialis anterior tendon sheath")` |
| H120 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G120,""<answer>(.*?)</answer>"") "),"Leigh syndrome")` |
| H121 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G121,""<answer>(.*?)</answer>"") "),"Malignant neoplasia")` |
| H122 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G122,""<answer>(.*?)</answer>"") "),"Breast cancer metastasis")` |
| H123 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G123,""<answer>(.*?)</answer>"") "),"OHVIRA / Herlyn–Werner–Wunderlich syndrome")` |
| H124 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G124,""<answer>(.*?)</answer>"") "),"Unicentric Castleman disease")` |
| H125 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G125,""<answer>(.*?)</answer>"") "),"Solitary fibrous tumour")` |
| H126 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G126,""<answer>(.*?)</answer>"") "),"Vesico-intestinal fistula")` |
| H127 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G127,""<answer>(.*?)</answer>"") "),"Usual interstitial pneumonia due to idiopathic pulmonary fibrosis")` |
| H128 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G128,""<answer>(.*?)</answer>"") "),"Spontaneous pneumothorax secondary to rheumatoid nodule")` |
| H129 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G129,""<answer>(.*?)</answer>"") "),"Linitis plastica in diffuse stomach carcinoma")` |
| H130 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G130,""<answer>(.*?)</answer>"") "),"Hepatocellular carcinoma")` |
| H131 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G131,""<answer>(.*?)</answer>"") "),"Acute Marchiafava–Bignami disease")` |
| H132 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G132,""<answer>(.*?)</answer>"") "),"Branchial cyst")` |
| H133 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G133,""<answer>(.*?)</answer>"") "),"Extraskeletal osteosarcoma")` |
| H134 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G134,""<answer>(.*?)</answer>"") "),"Cochlear hypoplasia type I (with aplasia of the cochlear nerve)")` |
| H135 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G135,""<answer>(.*?)</answer>"") "),"Xanthogranulomatous pyelonephritis")` |
| H136 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G136,""<answer>(.*?)</answer>"") "),"Left ventricular noncompaction")` |
| H137 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G137,""<answer>(.*?)</answer>"") "),"Infantile cortical hyperostosis (Caffey disease)")` |
| H138 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G138,""<answer>(.*?)</answer>"") "),"Split hand/split foot malformation (SHFM)")` |
| H139 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G139,""<answer>(.*?)</answer>"") "),"Huntington’s disease")` |
| H140 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G140,""<answer>(.*?)</answer>"") "),"Haemangioma")` |
| H141 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G141,""<answer>(.*?)</answer>"") "),"Medulloblastoma with extraneural metastases")` |
| H142 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G142,""<answer>(.*?)</answer>"") "),"Plasma cell mastitis")` |
| H143 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G143,""<answer>(.*?)</answer>"") "),"Leukoencephalopathy with brainstem and spinal cord involvement and lactate elevation (LBSL)")` |
| H144 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G144,""<answer>(.*?)</answer>"") "),"Hirayama disease")` |
| H145 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G145,""<answer>(.*?)</answer>"") "),"Cholangiocarcinoma")` |
| H146 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G146,""<answer>(.*?)</answer>"") "),"Autosplenectomy (rock spleen) secondary to sickle cell disease")` |
| H147 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G147,""<answer>(.*?)</answer>"") "),"Desmoid tumour")` |
| H148 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G148,""<answer>(.*?)</answer>"") "),"Müller–Weiss syndrome")` |
| H149 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G149,""<answer>(.*?)</answer>"") "),"Fibroadenoma")` |
| H150 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G150,""<answer>(.*?)</answer>"") "),"Paradoxical tuberculomas post-antituberculosis treatment")` |
| H151 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G151,""<answer>(.*?)</answer>"") "),"Caroli syndrome (congenital hepatic fibrosis with Caroli disease)")` |
| H152 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G152,""<answer>(.*?)</answer>"") "),"Pantothenate kinase-associated neurodegeneration")` |
| H153 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G153,""<answer>(.*?)</answer>"") "),"Langerhans cell histiocytosis")` |
| H154 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G154,""<answer>(.*?)</answer>"") "),"Giant microcystic serous cystadenoma of the pancreas")` |
| H155 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G155,""<answer>(.*?)</answer>"") "),"Malignant peripheral nerve sheath tumour")` |
| H156 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G156,""<answer>(.*?)</answer>"") "),"Disseminated AIDS-related Kaposi sarcoma")` |
| H157 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G157,""<answer>(.*?)</answer>"") "),"Neonatal mastitis with abscess")` |
| H158 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G158,""<answer>(.*?)</answer>"") "),"Fungal osteomyelitis of the patella")` |
| H159 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G159,""<answer>(.*?)</answer>"") "),"Cervical agenesis with endometrioma")` |
| H160 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G160,""<answer>(.*?)</answer>"") "),"Brachial plexus schwannomas")` |
| H161 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G161,""<answer>(.*?)</answer>"") "),"Uremic encephalopathy")` |
| H162 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G162,""<answer>(.*?)</answer>"") "),"Scrotal filariasis")` |
| H163 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G163,""<answer>(.*?)</answer>"") "),"Lymphoma")` |
| H164 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G164,""<answer>(.*?)</answer>"") "),"Infantile cortical hyperostosis (Caffey disease)")` |
| H165 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G165,""<answer>(.*?)</answer>"") "),"Split hand/split foot malformation (SHFM)")` |
| H166 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G166,""<answer>(.*?)</answer>"") "),"Spinal epidural abscess")` |
| H167 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G167,""<answer>(.*?)</answer>"") "),"Neurocutaneous melanosis")` |
| H168 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G168,""<answer>(.*?)</answer>"") "),"Cystic fibrosis")` |
| H169 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G169,""<answer>(.*?)</answer>"") "),"Osteoblastoma")` |
| H170 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G170,""<answer>(.*?)</answer>"") "),"Plexiform neurofibroma")` |
| H171 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G171,""<answer>(.*?)</answer>"") "),"Mycetoma foot")` |
| H172 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G172,""<answer>(.*?)</answer>"") "),"Aquaporin-4 IgG-positive neuromyelitis optica spectrum disorders (AQP4-NMOSD)")` |
| H173 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G173,""<answer>(.*?)</answer>"") "),"Retropancreatic mature teratoma")` |
| H174 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G174,""<answer>(.*?)</answer>"") "),"Retroperitoneal myxoid liposarcoma")` |
| H175 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G175,""<answer>(.*?)</answer>"") "),"Ependymoma")` |
| H176 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G176,""<answer>(.*?)</answer>"") "),"Angiomyolipoma")` |
| H177 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G177,""<answer>(.*?)</answer>"") "),"Dysgerminoma")` |
| H178 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G178,""<answer>(.*?)</answer>"") "),"Haematometrocolpos")` |
| H179 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G179,""<answer>(.*?)</answer>"") "),"Ewing sarcoma (or other small round blue cell tumour)")` |
| H180 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G180,""<answer>(.*?)</answer>"") "),"Spinal neurosarcoidosis with trident sign")` |
| H181 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G181,""<answer>(.*?)</answer>"") "),"Pyogenic spondylodiscitis")` |
| H182 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G182,""<answer>(.*?)</answer>"") "),"Hereditary haemorrhagic telangiectasia")` |
| H183 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G183,""<answer>(.*?)</answer>"") "),"Haemangioblastoma")` |
| H184 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G184,""<answer>(.*?)</answer>"") "),"First branchial cleft cyst")` |
| H185 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G185,""<answer>(.*?)</answer>"") "),"Acromegaly secondary to pituitary macroadenoma")` |
| H186 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G186,""<answer>(.*?)</answer>"") "),"Wallerian degeneration of pontocerebellar tracts secondary to unilateral pontine infarction")` |
| H187 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G187,""<answer>(.*?)</answer>"") "),"Stump appendicitis")` |
| H188 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G188,""<answer>(.*?)</answer>"") "),"Xanthogranulomatous pyelonephritis (diffuse form)")` |
| H189 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G189,""<answer>(.*?)</answer>"") "),"Adrenal adenocarcinoma")` |
| H190 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G190,""<answer>(.*?)</answer>"") "),"Classical osteosarcoma")` |
| H191 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G191,""<answer>(.*?)</answer>"") "),"Primary prostatic urethral neoplasm")` |
| H192 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G192,""<answer>(.*?)</answer>"") "),"Pyogenic arthritis")` |
| H193 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G193,""<answer>(.*?)</answer>"") "),"Anaplastic meningioma (WHO Grade III)")` |
| H194 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G194,""<answer>(.*?)</answer>"") "),"Pott’s spine with psoas abscess")` |
| H195 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G195,""<answer>(.*?)</answer>"") "),"Intramuscular haemangioma")` |
| H196 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G196,""<answer>(.*?)</answer>"") "),"Congenital femoral deficiency and absence of the anterior cruciate ligament")` |
| H197 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G197,""<answer>(.*?)</answer>"") "),"Neuropathic arthropathy")` |
| H198 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G198,""<answer>(.*?)</answer>"") "),"α-dystroglycanopathies (and Walker–Warburg syndrome)")` |
| H199 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G199,""<answer>(.*?)</answer>"") "),"Emphysematous hepatitis")` |
| H200 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G200,""<answer>(.*?)</answer>"") "),"Progressive fibrodysplasia ossificans")` |
| H201 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G201,""<answer>(.*?)</answer>"") "),"Multiple hereditary exostoses")` |
| H202 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G202,""<answer>(.*?)</answer>"") "),"Sinding–Larsen–Johansson disease (SLJD)")` |
| H203 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G203,""<answer>(.*?)</answer>"") "),"Haberland syndrome (Encephalocraniocutaneous lipomatosis)")` |
| H204 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G204,""<answer>(.*?)</answer>"") "),"Pilocytic astrocytoma")` |
| H205 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G205,""<answer>(.*?)</answer>"") "),"Probable Creutzfeldt–Jakob disease")` |
| H206 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G206,""<answer>(.*?)</answer>"") "),"Lipoidal proteinosis")` |
| H207 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G207,""<answer>(.*?)</answer>"") "),"Rheumatoid arthritis")` |
| H208 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G208,""<answer>(.*?)</answer>"") "),"Giant caseating cerebellar tuberculoma")` |
| H209 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G209,""<answer>(.*?)</answer>"") "),"Sturge–Weber syndrome")` |
| H210 | `=IFERROR(__xludf.DUMMYFUNCTION("REGEXEXTRACT(G210,""<answer>(.*?)</answer>"") "),"Cochlear nerve deficiency")` |

