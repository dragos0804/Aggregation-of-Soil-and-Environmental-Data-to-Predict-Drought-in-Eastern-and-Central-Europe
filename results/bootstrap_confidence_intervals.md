# Location-Block Bootstrap Confidence Intervals

Location-block bootstrap (1000 resamples of unique (LAT, LON) test locations, sampled with replacement) applied to the primary SPEI-01, threshold -1.0, common-support RF/SVM held-out predictions. Resampling whole locations rather than individual rows preserves within-location month-to-month dependence. These intervals are distinct from the three-fold spatial/temporal extrapolation normal-approximation intervals reported elsewhere in the manuscript, which summarize variability across geographic/temporal blocks rather than sampling uncertainty of a single held-out set.

The Random Forest intervals use the full 237,132-row held-out set. The SVM intervals use the retained resource-bounded run with 5,000 training rows and 5,000 held-out evaluation rows.

|   data_start_year |   data_end_year |   svm_subsample_size | pipeline                              | model            | metric      |   point_estimate |   ci_lower_95 |   ci_upper_95 |   n_bootstrap |   n_locations |   n_rows |
|------------------:|----------------:|---------------------:|:--------------------------------------|:-----------------|:------------|-----------------:|--------------:|--------------:|--------------:|--------------:|---------:|
|              1990 |            2024 |                50000 | NASA-Only (common support)            | Random Forest    | drought_f1  |           0.1183 |        0.1147 |        0.1219 |          1000 |          2823 |   237132 |
|              1990 |            2024 |                50000 | NASA-Only (common support)            | Random Forest    | weighted_f1 |           0.6641 |        0.6608 |        0.6673 |          1000 |          2823 |   237132 |
|              1990 |            2024 |                 5000 | NASA-Only (common support)            | SVM (RBF Kernel) | drought_f1  |           0.4007 |        0.3800 |        0.4227 |          1000 |          2341 |     5000 |
|              1990 |            2024 |                 5000 | NASA-Only (common support)            | SVM (RBF Kernel) | weighted_f1 |           0.6079 |        0.5926 |        0.6227 |          1000 |          2341 |     5000 |
|              1990 |            2024 |                50000 | Fused (SOTWIS + NASA, common support) | Random Forest    | drought_f1  |           0.1531 |        0.1493 |        0.1571 |          1000 |          2823 |   237132 |
|              1990 |            2024 |                50000 | Fused (SOTWIS + NASA, common support) | Random Forest    | weighted_f1 |           0.6654 |        0.6620 |        0.6687 |          1000 |          2823 |   237132 |
|              1990 |            2024 |                 5000 | Fused (SOTWIS + NASA, common support) | SVM (RBF Kernel) | drought_f1  |           0.3781 |        0.3558 |        0.4014 |          1000 |          2341 |     5000 |
|              1990 |            2024 |                 5000 | Fused (SOTWIS + NASA, common support) | SVM (RBF Kernel) | weighted_f1 |           0.6358 |        0.6209 |        0.6509 |          1000 |          2341 |     5000 |
