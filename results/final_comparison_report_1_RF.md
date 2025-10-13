# Final Comparison of Model Results

This report compares the performance of a Random Forest model trained on three different feature sets to determine the value of data fusion for drought prediction.

The key metrics are:
1.  **Drought F1-Score:** The F1-score for the positive class ('Drought'). This is the most important metric for evaluating performance on the rare but critical drought events.
2.  **Weighted Avg F1-Score:** The overall F1-score, weighted by the number of samples in each class.

|                       |   Drought F1-Score |   Weighted Avg F1-Score |
|:----------------------|-------------------:|------------------------:|
| SOTWIS-Only           |             0.3655 |                  0.5815 |
| NASA-Only             |             0.1692 |                  0.7374 |
| Fused (SOTWIS + NASA) |             0.1857 |                  0.7485 |