# Canonical Model/Dataset Confusion-Matrix Metrics

| Model            | Dataset                                                 |     TN |     FP |    FN |     TP |   Precision_D |   Recall_D |   F1_D |
|:-----------------|:--------------------------------------------------------|-------:|-------:|------:|-------:|--------------:|-----------:|-------:|
| Random Forest    | SOTWIS-Only (separate static susceptibility experiment) | 370255 | 199228 | 83093 |  79988 |        0.2865 |     0.4905 | 0.3617 |
| Random Forest    | NASA-Only (common support)                              | 166672 |  13999 | 52030 |   4431 |        0.2404 |     0.0785 | 0.1183 |
| Random Forest    | Fused (SOTWIS + NASA, common support)                   | 162285 |  18386 | 50255 |   6206 |        0.2524 |     0.1099 | 0.1531 |
| SVM (RBF Kernel) | SOTWIS-Only (separate static susceptibility experiment) |  56023 |  21711 | 13441 |   8825 |        0.289  |     0.3963 | 0.3343 |
| SVM (RBF Kernel) | NASA-Only (common support)                              |  40206 |  36121 |  8220 |  15453 |        0.2996 |     0.6528 | 0.4107 |
| SVM (RBF Kernel) | Fused (SOTWIS + NASA, common support)                   |  43022 |  33305 |  8974 |  14699 |        0.3062 |     0.6209 | 0.4101 |
| LSTM             | SOTWIS-Only (separate static susceptibility experiment) | 277577 | 283814 | 55917 | 106535 |        0.2729 |     0.6558 | 0.3854 |
| LSTM             | NASA-Only (common support)                              | 113807 |  64198 | 30172 |  26132 |        0.2893 |     0.4641 | 0.3564 |
| LSTM             | Fused (SOTWIS + NASA, common support)                   | 109137 |  68868 | 27565 |  28739 |        0.2944 |     0.5104 | 0.3734 |
