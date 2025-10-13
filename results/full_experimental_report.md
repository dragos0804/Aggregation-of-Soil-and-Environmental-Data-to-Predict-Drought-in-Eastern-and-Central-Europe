# Full Experimental Results: Drought Class Prediction

This report compares the performance of three models across three different feature sets. All metrics reported are for the minority 'Drought' class (SPEI < -1.0).

## Table 1: Precision for 'Drought' Class
*(What percentage of drought predictions were correct?)*

|                  |   SOTWIS-Only |   NASA-Only |   Fused (SOTWIS + NASA) |
|:-----------------|--------------:|------------:|------------------------:|
| Random Forest    |        0.2521 |      0.2457 |                  0.3529 |
| SVM (RBF Kernel) |        0.263  |      0.2709 |                  0.2847 |
| LSTM             |        0.1154 |      0.2516 |                  0.2686 |

## Table 2: Recall for 'Drought' Class
*(What percentage of actual droughts were found?)*

|                  |   SOTWIS-Only |   NASA-Only |   Fused (SOTWIS + NASA) |
|:-----------------|--------------:|------------:|------------------------:|
| Random Forest    |        0.6631 |      0.1302 |                  0.1284 |
| SVM (RBF Kernel) |        0.4619 |      0.645  |                  0.561  |
| LSTM             |        0      |      0.5704 |                  0.6001 |

## Table 3: F1-Score for 'Drought' Class
*(The harmonic mean of Precision and Recall, representing a balanced score.)*

|                  |   SOTWIS-Only |   NASA-Only |   Fused (SOTWIS + NASA) |
|:-----------------|--------------:|------------:|------------------------:|
| Random Forest    |        0.3653 |      0.1702 |                  0.1883 |
| SVM (RBF Kernel) |        0.3352 |      0.3815 |                  0.3777 |
| LSTM             |        0      |      0.3491 |                  0.3711 |
