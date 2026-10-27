# Full Experimental Results: Drought Class Prediction

NASA monthly missing values use the `median_indicators` strategy; medians and missingness indicators, when selected, are fitted on the chronological training partition only.

This report compares the performance of three models across three different feature sets. Drought-class precision, recall, F1, and PR-AUC are reported for the minority class (SPEI-01 < -1); weighted F1 is a complementary class-imbalance-aware summary.

All models predict the drought class for the next month using features available through the current month. LSTM sequences are restricted to one location and consecutive monthly observations.

## Table 1: Precision for 'Drought' Class
*(What percentage of drought predictions were correct?)*

|                  |   SOTWIS-Only (separate static susceptibility experiment) |   NASA-Only (common support) |   Fused (SOTWIS + NASA, common support) |
|:-----------------|----------------------------------------------------------:|-----------------------------:|----------------------------------------:|
| Random Forest    |                                                    0.2865 |                       0.2404 |                                  0.2524 |
| SVM (RBF Kernel) |                                                    0.289  |                       0.2996 |                                  0.3062 |
| LSTM             |                                                    0.2729 |                       0.2893 |                                  0.2944 |

## Table 2: Recall for 'Drought' Class
*(What percentage of actual droughts were found?)*

|                  |   SOTWIS-Only (separate static susceptibility experiment) |   NASA-Only (common support) |   Fused (SOTWIS + NASA, common support) |
|:-----------------|----------------------------------------------------------:|-----------------------------:|----------------------------------------:|
| Random Forest    |                                                    0.4905 |                       0.0785 |                                  0.1099 |
| SVM (RBF Kernel) |                                                    0.3963 |                       0.6528 |                                  0.6209 |
| LSTM             |                                                    0.6558 |                       0.4641 |                                  0.5104 |

## Table 3: F1-Score for 'Drought' Class
*(The harmonic mean of Precision and Recall, representing a balanced score.)*

|                  |   SOTWIS-Only (separate static susceptibility experiment) |   NASA-Only (common support) |   Fused (SOTWIS + NASA, common support) |
|:-----------------|----------------------------------------------------------:|-----------------------------:|----------------------------------------:|
| Random Forest    |                                                    0.3617 |                       0.1183 |                                  0.1531 |
| SVM (RBF Kernel) |                                                    0.3343 |                       0.4107 |                                  0.4101 |
| LSTM             |                                                    0.3854 |                       0.3564 |                                  0.3734 |

## Table 4: Weighted F1-Score
*(Overall F1 weighted by class support, useful for operational reliability under imbalance.)*

|                  |   SOTWIS-Only (separate static susceptibility experiment) |   NASA-Only (common support) |   Fused (SOTWIS + NASA, common support) |
|:-----------------|----------------------------------------------------------:|-----------------------------:|----------------------------------------:|
| Random Forest    |                                                    0.6433 |                       0.6641 |                                  0.6654 |
| SVM (RBF Kernel) |                                                    0.6661 |                       0.5892 |                                  0.6089 |
| LSTM             |                                                    0.5676 |                       0.6227 |                                  0.6167 |

## Table 5: Average 'Drought' F1-Score by Dataset

|                                                         |   Average Drought F1-Score |
|:--------------------------------------------------------|---------------------------:|
| SOTWIS-Only (separate static susceptibility experiment) |                     0.3605 |
| NASA-Only (common support)                              |                     0.2952 |
| Fused (SOTWIS + NASA, common support)                   |                     0.3122 |

## Table 6: Average Weighted F1-Score by Dataset

|                                                         |   Average Weighted F1-Score |
|:--------------------------------------------------------|----------------------------:|
| SOTWIS-Only (separate static susceptibility experiment) |                      0.6257 |
| NASA-Only (common support)                              |                      0.6253 |
| Fused (SOTWIS + NASA, common support)                   |                      0.6303 |

## Detailed Confusion Matrices and Per-Class Metrics

### Random Forest on SOTWIS-Only (separate static susceptibility experiment)

Confusion matrix:

|                   |   Pred No Drought |   Pred Drought |
|:------------------|------------------:|---------------:|
| Actual No Drought |            370255 |         199228 |
| Actual Drought    |             83093 |          79988 |

Per-class metrics:

| Class      |   Precision |   Recall |   F1-Score |   Support |
|:-----------|------------:|---------:|-----------:|----------:|
| No Drought |      0.8167 |   0.6502 |     0.724  |    569483 |
| Drought    |      0.2865 |   0.4905 |     0.3617 |    163081 |

### Random Forest on NASA-Only (common support)

Confusion matrix:

|                   |   Pred No Drought |   Pred Drought |
|:------------------|------------------:|---------------:|
| Actual No Drought |            166672 |          13999 |
| Actual Drought    |             52030 |           4431 |

Per-class metrics:

| Class      |   Precision |   Recall |   F1-Score |   Support |
|:-----------|------------:|---------:|-----------:|----------:|
| No Drought |      0.7621 |   0.9225 |     0.8347 |    180671 |
| Drought    |      0.2404 |   0.0785 |     0.1183 |     56461 |

### Random Forest on Fused (SOTWIS + NASA, common support)

Confusion matrix:

|                   |   Pred No Drought |   Pred Drought |
|:------------------|------------------:|---------------:|
| Actual No Drought |            162285 |          18386 |
| Actual Drought    |             50255 |           6206 |

Per-class metrics:

| Class      |   Precision |   Recall |   F1-Score |   Support |
|:-----------|------------:|---------:|-----------:|----------:|
| No Drought |      0.7636 |   0.8982 |     0.8254 |    180671 |
| Drought    |      0.2524 |   0.1099 |     0.1531 |     56461 |

### SVM (RBF Kernel) on SOTWIS-Only (separate static susceptibility experiment)

Confusion matrix:

|                   |   Pred No Drought |   Pred Drought |
|:------------------|------------------:|---------------:|
| Actual No Drought |             56023 |          21711 |
| Actual Drought    |             13441 |           8825 |

Per-class metrics:

| Class      |   Precision |   Recall |   F1-Score |   Support |
|:-----------|------------:|---------:|-----------:|----------:|
| No Drought |      0.8065 |   0.7207 |     0.7612 |     77734 |
| Drought    |      0.289  |   0.3963 |     0.3343 |     22266 |

### SVM (RBF Kernel) on NASA-Only (common support)

Confusion matrix:

|                   |   Pred No Drought |   Pred Drought |
|:------------------|------------------:|---------------:|
| Actual No Drought |             40206 |          36121 |
| Actual Drought    |              8220 |          15453 |

Per-class metrics:

| Class      |   Precision |   Recall |   F1-Score |   Support |
|:-----------|------------:|---------:|-----------:|----------:|
| No Drought |      0.8303 |   0.5268 |     0.6446 |     76327 |
| Drought    |      0.2996 |   0.6528 |     0.4107 |     23673 |

### SVM (RBF Kernel) on Fused (SOTWIS + NASA, common support)

Confusion matrix:

|                   |   Pred No Drought |   Pred Drought |
|:------------------|------------------:|---------------:|
| Actual No Drought |             43022 |          33305 |
| Actual Drought    |              8974 |          14699 |

Per-class metrics:

| Class      |   Precision |   Recall |   F1-Score |   Support |
|:-----------|------------:|---------:|-----------:|----------:|
| No Drought |      0.8274 |   0.5637 |     0.6705 |     76327 |
| Drought    |      0.3062 |   0.6209 |     0.4101 |     23673 |

### LSTM on SOTWIS-Only (separate static susceptibility experiment)

Confusion matrix:

|                   |   Pred No Drought |   Pred Drought |
|:------------------|------------------:|---------------:|
| Actual No Drought |            277577 |         283814 |
| Actual Drought    |             55917 |         106535 |

Per-class metrics:

| Class      |   Precision |   Recall |   F1-Score |   Support |
|:-----------|------------:|---------:|-----------:|----------:|
| No Drought |      0.8323 |   0.4944 |     0.6204 |    561391 |
| Drought    |      0.2729 |   0.6558 |     0.3854 |    162452 |

### LSTM on NASA-Only (common support)

Confusion matrix:

|                   |   Pred No Drought |   Pred Drought |
|:------------------|------------------:|---------------:|
| Actual No Drought |            113807 |          64198 |
| Actual Drought    |             30172 |          26132 |

Per-class metrics:

| Class      |   Precision |   Recall |   F1-Score |   Support |
|:-----------|------------:|---------:|-----------:|----------:|
| No Drought |      0.7904 |   0.6393 |     0.7069 |    178005 |
| Drought    |      0.2893 |   0.4641 |     0.3564 |     56304 |

### LSTM on Fused (SOTWIS + NASA, common support)

Confusion matrix:

|                   |   Pred No Drought |   Pred Drought |
|:------------------|------------------:|---------------:|
| Actual No Drought |            109137 |          68868 |
| Actual Drought    |             27565 |          28739 |

Per-class metrics:

| Class      |   Precision |   Recall |   F1-Score |   Support |
|:-----------|------------:|---------:|-----------:|----------:|
| No Drought |      0.7984 |   0.6131 |     0.6936 |    178005 |
| Drought    |      0.2944 |   0.5104 |     0.3734 |     56304 |
