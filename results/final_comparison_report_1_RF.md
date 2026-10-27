# Final Comparison of Model Results

This report compares the performance of a Random Forest model trained on three different feature sets to determine the value of data fusion for drought prediction.

The key metrics are:
1.  **Drought F1-Score:** The F1-score for the positive class ('Drought'). This is the most important metric for evaluating performance on the rare but critical drought events.
2.  **Weighted Avg F1-Score:** The overall F1-score, weighted by the number of samples in each class.

|                       |   Drought F1-Score |   Weighted Avg F1-Score |
|:----------------------|-------------------:|------------------------:|
| SOTWIS-Only           |             0.3574 |                  0.5942 |
| NASA-Only             |             0.3456 |                  0.7438 |
| Fused (SOTWIS + NASA) |             0.3313 |                  0.7946 |

The regenerated run changes the interpretation materially. The NASA-Only Random Forest is no longer weak on the drought class; it now reaches a drought F1-score of 0.3456 while preserving stronger overall reliability than the soil-only baseline. The fused Random Forest remains the most reliable of the three in weighted terms, but its drought-class F1-score stays below the NASA-Only configuration, indicating that simple feature fusion does not automatically improve minority-class detection for the tree-based model.