# Phase-3 sweep aggregate

runs: 6

| model                                      | channels   | task      |   n_seeds | direction_macro_f1   | macro_f1        | transition_f1   | non_transition_f1   | worst_direction_f1   | accuracy        | inference_ms_per_window   | params              |
|:-------------------------------------------|:-----------|:----------|----------:|:---------------------|:----------------|:----------------|:--------------------|:---------------------|:----------------|:--------------------------|:--------------------|
| complex_selective + selective_gyrophase_v2 | acc_gyro   | direction |         3 | 0.6365 ± 0.0826      | 0.6872 ± 0.0712 | 0.8470 ± 0.0464 | 0.9912 ± 0.0029     | 0.4944 ± 0.0841      | 0.9705 ± 0.0063 | 0.1551 ± 0.0009           | 51254.0000 ± 0.0000 |
| complex_selective + selective_gyrophase_v3 | acc_gyro   | direction |         3 | 0.5580 ± 0.0412      | 0.6179 ± 0.0367 | 0.7018 ± 0.1214 | 0.9778 ± 0.0133     | 0.3520 ± 0.1671      | 0.9455 ± 0.0255 | 0.1547 ± 0.0006           | 51254.0000 ± 0.0000 |
