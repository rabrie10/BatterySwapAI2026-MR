# Transfer stress: leave-building(s)-out at scenario cutoffs

## Verdict

- (a) PR-AUC on 5 hard holdouts: cens mean 0.428 vs v7 0.391 (cens better on 5/5).
- (a) rank PR-AUC on 5 hard holdouts: cens mean 0.375 vs v7 0.370 (cens better on 2/5).
- (b) v7: in-sample raw 8.717; 5-fold OOF raw 8.45; LOO raw 9.946 (x1.052), LOO calibrated 10.215 (x1.08); worst hard-fold calibrated x1.129.
- (b) cens: in-sample raw 12.37; 5-fold OOF raw 10.01; LOO raw 12.265 (x1.297), LOO calibrated 10.029 (x1.06); worst hard-fold calibrated x1.211.
- (c) v7 at equal volume k=454: global-by-level precision 0.4031, per-scenario quota 0.3722; building-volume Spearman 0.752; top-bucket >=0.7 realized 0.357 on n=238.
- (c) cens at equal volume k=454: global-by-level precision 0.3943, per-scenario quota 0.3767; building-volume Spearman 0.818; top-bucket >=0.7 realized 0.462 on n=182.
- (d) per-building median dispersion: beta_30 CV 0.469 (max/min 5.79x) vs beta_rise CV 0.054 (max/min 1.32x); v_std_30 CV 0.682 vs v_std_rise CV 0.066. Substitute the within-day SCALE features with their rise ratios.

Both target variants refit per fold on windows excluding the held-out buildings (stride 8, max_iter 150); all metrics are on held-out buildings' scenario rows only. `cal` = RemainingCalibration fitted on the fold's training buildings, i.e. the production procedure applied to a fresh building.

## Reconstruction check (shipped artifacts over the rebuilt scenario frame)

```json
{
  "v7": {
    "rows": 19890,
    "sum_p_raw_per_scenario": 8.717,
    "sum_p_cal_per_scenario": 8.737,
    "realized_per_scenario": 9.458
  },
  "cens": {
    "rows": 19890,
    "sum_p_raw_per_scenario": 12.37,
    "sum_p_cal_per_scenario": 10.163,
    "realized_per_scenario": 9.458
  }
}
```

## Hard grouped holdouts

| fold | held-out (dev/EOL) | variant | sum p/scen raw | cal | realized | infl raw | infl cal | PR-AUC lvl | PR-AUC rank | P@5 lvl | P@5 rank | P@10 lvl | P@10 rank | top>=0.7 cal (n, realized) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hard_large5 | 292/32 | v7 | 2.271 | 2.305 | 3.729 | 0.609 | 0.618 | 0.4643 | 0.4238 | 1.0 | 1.0 | 0.9 | 1.0 | 29, 0.828 |
| hard_large5 | 292/32 | cens | 4.355 | 3.592 | 3.729 | 1.168 | 0.963 | 0.4968 | 0.4402 | 1.0 | 1.0 | 0.9 | 0.8 | 51, 0.667 |
| hard_small10 | 36/16 | v7 | 1.045 | 1.195 | 1.833 | 0.57 | 0.652 | 0.5529 | 0.4239 | 1.0 | 1.0 | 0.9 | 0.8 | 30, 0.633 |
| hard_small10 | 36/16 | cens | 1.214 | 1.057 | 1.833 | 0.662 | 0.576 | 0.6026 | 0.4155 | 1.0 | 1.0 | 1.0 | 0.9 | 25, 0.68 |
| hard_mosteol5 | 168/45 | v7 | 5.34 | 5.88 | 5.208 | 1.025 | 1.129 | 0.2664 | 0.313 | 1.0 | 1.0 | 0.5 | 0.5 | 155, 0.284 |
| hard_mosteol5 | 168/45 | cens | 6.585 | 6.309 | 5.208 | 1.264 | 1.211 | 0.2879 | 0.3125 | 1.0 | 1.0 | 0.7 | 0.7 | 144, 0.271 |
| hard_hirate6 | 42/29 | v7 | 2.917 | 3.124 | 3.312 | 0.881 | 0.943 | 0.3214 | 0.3123 | 0.8 | 0.8 | 0.6 | 0.6 | 108, 0.37 |
| hard_hirate6 | 42/29 | cens | 3.32 | 2.971 | 3.312 | 1.002 | 0.897 | 0.3387 | 0.3106 | 1.0 | 1.0 | 0.6 | 0.5 | 77, 0.494 |
| hard_betashift5 | 38/21 | v7 | 1.996 | 2.008 | 2.479 | 0.805 | 0.81 | 0.3521 | 0.3772 | 0.6 | 0.6 | 0.3 | 0.3 | 41, 0.439 |
| hard_betashift5 | 38/21 | cens | 2.283 | 1.876 | 2.479 | 0.921 | 0.757 | 0.4129 | 0.3954 | 0.8 | 0.8 | 0.4 | 0.4 | 24, 0.75 |

### Policies on held-out rows (level threshold vs rank+quota)

| fold | variant | policy | swaps/scen | precision | recall |
|---|---|---|---|---|---|
| hard_large5 | v7 | level_tau_0.35 | 2.083 | 0.58 | 0.324 |
| hard_large5 | v7 | level_tau_0.5 | 1.312 | 0.6984 | 0.2458 |
| hard_large5 | v7 | level_tau_train_matched | 1.417 | 0.6765 | 0.257 |
| hard_large5 | v7 | rank_quota_train_rate | 11.417 | 0.2646 | 0.8101 |
| hard_large5 | cens | level_tau_0.35 | 3.958 | 0.4684 | 0.4972 |
| hard_large5 | cens | level_tau_0.5 | 2.25 | 0.5926 | 0.3575 |
| hard_large5 | cens | level_tau_train_matched | 1.938 | 0.6237 | 0.324 |
| hard_large5 | cens | rank_quota_train_rate | 11.417 | 0.2609 | 0.7989 |
| hard_small10 | v7 | level_tau_0.35 | 1.062 | 0.5882 | 0.3409 |
| hard_small10 | v7 | level_tau_0.5 | 0.75 | 0.6111 | 0.25 |
| hard_small10 | v7 | level_tau_train_matched | 1.125 | 0.5926 | 0.3636 |
| hard_small10 | v7 | rank_quota_train_rate | 0.562 | 0.5556 | 0.1705 |
| hard_small10 | cens | level_tau_0.35 | 1.0 | 0.6458 | 0.3523 |
| hard_small10 | cens | level_tau_0.5 | 0.688 | 0.6667 | 0.25 |
| hard_small10 | cens | level_tau_train_matched | 0.979 | 0.6383 | 0.3409 |
| hard_small10 | cens | rank_quota_train_rate | 0.562 | 0.5556 | 0.1705 |
| hard_mosteol5 | v7 | level_tau_0.35 | 6.0 | 0.3056 | 0.352 |
| hard_mosteol5 | v7 | level_tau_0.5 | 4.125 | 0.3333 | 0.264 |
| hard_mosteol5 | v7 | level_tau_train_matched | 4.75 | 0.3377 | 0.308 |
| hard_mosteol5 | v7 | rank_quota_train_rate | 2.146 | 0.3786 | 0.156 |
| hard_mosteol5 | cens | level_tau_0.35 | 7.062 | 0.3245 | 0.44 |
| hard_mosteol5 | cens | level_tau_0.5 | 4.875 | 0.3205 | 0.3 |
| hard_mosteol5 | cens | level_tau_train_matched | 5.625 | 0.3407 | 0.368 |
| hard_mosteol5 | cens | rank_quota_train_rate | 2.146 | 0.3592 | 0.148 |
| hard_hirate6 | v7 | level_tau_0.35 | 3.292 | 0.3797 | 0.3774 |
| hard_hirate6 | v7 | level_tau_0.5 | 2.917 | 0.3643 | 0.3208 |
| hard_hirate6 | v7 | level_tau_train_matched | 3.354 | 0.3913 | 0.3962 |
| hard_hirate6 | v7 | rank_quota_train_rate | 0.188 | 0.4444 | 0.0252 |
| hard_hirate6 | cens | level_tau_0.35 | 3.417 | 0.4146 | 0.4277 |
| hard_hirate6 | cens | level_tau_0.5 | 2.854 | 0.3942 | 0.3396 |
| hard_hirate6 | cens | level_tau_train_matched | 3.375 | 0.4198 | 0.4277 |
| hard_hirate6 | cens | rank_quota_train_rate | 0.188 | 0.4444 | 0.0252 |
| hard_betashift5 | v7 | level_tau_0.35 | 2.062 | 0.404 | 0.3361 |
| hard_betashift5 | v7 | level_tau_0.5 | 1.458 | 0.4 | 0.2353 |
| hard_betashift5 | v7 | level_tau_train_matched | 2.167 | 0.4038 | 0.3529 |
| hard_betashift5 | v7 | rank_quota_train_rate | 0.292 | 0.9286 | 0.1092 |
| hard_betashift5 | cens | level_tau_0.35 | 1.979 | 0.4947 | 0.395 |
| hard_betashift5 | cens | level_tau_0.5 | 1.375 | 0.5303 | 0.2941 |
| hard_betashift5 | cens | level_tau_train_matched | 1.917 | 0.5 | 0.3866 |
| hard_betashift5 | cens | rank_quota_train_rate | 0.292 | 1.0 | 0.1176 |

## Pooled leave-one-building-out (24 folds, every row out-of-building)

5-fold-by-building OOF reference (the validation the team used):
```json
{
  "v7": {
    "sum_p_raw_per_scenario": 8.45,
    "realized_per_scenario": 9.46,
    "inflation_raw": 0.893
  },
  "cens": {
    "sum_p_raw_per_scenario": 10.01,
    "realized_per_scenario": 9.46,
    "inflation_raw": 1.058
  }
}
```

### v7
```json
{
  "n_rows": 19890,
  "n_due": 454,
  "sum_p_raw_per_scenario": 9.946,
  "sum_p_cal_per_scenario": 10.215,
  "realized_per_scenario": 9.458,
  "inflation_raw": 1.052,
  "inflation_cal": 1.08,
  "pr_auc_level": 0.2796,
  "pr_auc_rank": 0.29,
  "precision_at_k_level": {
    "25": 0.28,
    "50": 0.14,
    "100": 0.19
  },
  "precision_at_k_rank": {
    "25": 0.28,
    "50": 0.24,
    "100": 0.25
  },
  "blocks_cal": {
    "early": {
      "predicted_per_scenario": 11.47,
      "realized_per_scenario": 13.25,
      "ratio": 0.866
    },
    "mid": {
      "predicted_per_scenario": 10.0,
      "realized_per_scenario": 8.56,
      "ratio": 1.168
    },
    "late": {
      "predicted_per_scenario": 9.17,
      "realized_per_scenario": 6.56,
      "ratio": 1.398
    }
  },
  "top_bucket_raw": {
    "n": 223,
    "predicted_mean": 0.883,
    "realized_mean": 0.287
  },
  "top_bucket_cal": {
    "n": 238,
    "predicted_mean": 0.898,
    "realized_mean": 0.357
  },
  "building_volume_spearman": 0.752,
  "equal_volume_selectors": {
    "k_total": 454,
    "global_by_level": {
      "precision": 0.4031,
      "recall": 0.4031
    },
    "per_scenario_quota": {
      "quota": 9,
      "precision": 0.3722,
      "recall": 0.3722
    }
  }
}
```

worst buildings by calibrated inflation: b_ad4a5872f2db x3.02 (34 due), b_955300d5f090 x2.6 (30 due), b_53df020ad893 x2.34 (22 due), b_4217a3b6e58b x2.15 (6 due), b_15bc00efc223 x1.53 (6 due), b_b7e9d5634ad3 x1.46 (10 due)

### cens
```json
{
  "n_rows": 19890,
  "n_due": 454,
  "sum_p_raw_per_scenario": 12.265,
  "sum_p_cal_per_scenario": 10.029,
  "realized_per_scenario": 9.458,
  "inflation_raw": 1.297,
  "inflation_cal": 1.06,
  "pr_auc_level": 0.3155,
  "pr_auc_rank": 0.2989,
  "precision_at_k_level": {
    "25": 0.44,
    "50": 0.24,
    "100": 0.15
  },
  "precision_at_k_rank": {
    "25": 0.32,
    "50": 0.24,
    "100": 0.26
  },
  "blocks_cal": {
    "early": {
      "predicted_per_scenario": 12.5,
      "realized_per_scenario": 13.25,
      "ratio": 0.944
    },
    "mid": {
      "predicted_per_scenario": 9.04,
      "realized_per_scenario": 8.56,
      "ratio": 1.056
    },
    "late": {
      "predicted_per_scenario": 8.54,
      "realized_per_scenario": 6.56,
      "ratio": 1.302
    }
  },
  "top_bucket_raw": {
    "n": 299,
    "predicted_mean": 0.898,
    "realized_mean": 0.381
  },
  "top_bucket_cal": {
    "n": 182,
    "predicted_mean": 0.882,
    "realized_mean": 0.462
  },
  "building_volume_spearman": 0.818,
  "equal_volume_selectors": {
    "k_total": 454,
    "global_by_level": {
      "precision": 0.3943,
      "recall": 0.3943
    },
    "per_scenario_quota": {
      "quota": 9,
      "precision": 0.3767,
      "recall": 0.3767
    }
  }
}
```

worst buildings by calibrated inflation: b_ad4a5872f2db x3.15 (34 due), b_955300d5f090 x2.31 (30 due), b_53df020ad893 x2.19 (22 due), b_4217a3b6e58b x1.57 (6 due), b_b7e9d5634ad3 x1.22 (10 due), b_15bc00efc223 x1.12 (6 due)

### Per-building LOO (calibrated sum p vs realized, per scenario)

| building | variant | rows | due | sum p cal/scen | realized/scen | infl cal | PR-AUC lvl | PR-AUC rank |
|---|---|---|---|---|---|---|---|---|
| b_15bc00efc223 | v7 | 219 | 6 | 0.192 | 0.125 | 1.534 | 0.5264 | 0.125 |
| b_15bc00efc223 | cens | 219 | 6 | 0.139 | 0.125 | 1.116 | 0.3122 | 0.125 |
| b_2b69c4007582 | v7 | 3036 | 36 | 0.205 | 0.75 | 0.274 | 0.6807 | 0.3325 |
| b_2b69c4007582 | cens | 3036 | 36 | 0.172 | 0.75 | 0.23 | 0.6723 | 0.3181 |
| b_4217a3b6e58b | v7 | 1075 | 6 | 0.268 | 0.125 | 2.145 | 0.594 | 0.125 |
| b_4217a3b6e58b | cens | 1075 | 6 | 0.196 | 0.125 | 1.572 | 0.7345 | 0.125 |
| b_53df020ad893 | v7 | 96 | 22 | 1.074 | 0.458 | 2.344 | 0.1727 | 0.1842 |
| b_53df020ad893 | cens | 96 | 22 | 1.003 | 0.458 | 2.189 | 0.2653 | 0.1836 |
| b_55038c6689a0 | v7 | 456 | 6 | 0.137 | 0.125 | 1.092 | 0.3682 | 0.0424 |
| b_55038c6689a0 | cens | 456 | 6 | 0.069 | 0.125 | 0.551 | 0.512 | 0.0399 |
| b_60c6d1796ab4 | v7 | 199 | 6 | 0.027 | 0.125 | 0.213 | 0.9762 | 0.1875 |
| b_60c6d1796ab4 | cens | 199 | 6 | 0.043 | 0.125 | 0.34 | 0.9444 | 0.1538 |
| b_62e09ab003f8 | v7 | 1386 | 40 | 0.543 | 0.833 | 0.651 | 0.6536 | 0.4226 |
| b_62e09ab003f8 | cens | 1386 | 40 | 0.728 | 0.833 | 0.874 | 0.6164 | 0.3839 |
| b_6869b0f15a78 | v7 | 518 | 6 | 0.025 | 0.125 | 0.203 | 0.9583 | 0.1304 |
| b_6869b0f15a78 | cens | 518 | 6 | 0.029 | 0.125 | 0.236 | 0.8593 | 0.1333 |
| b_7aac45259c8a | v7 | 181 | 6 | 0.036 | 0.125 | 0.289 | 0.9444 | 0.1277 |
| b_7aac45259c8a | cens | 181 | 6 | 0.026 | 0.125 | 0.204 | 0.9167 | 0.1304 |
| b_955300d5f090 | v7 | 5554 | 30 | 1.625 | 0.625 | 2.6 | 0.3156 | 0.2372 |
| b_955300d5f090 | cens | 5554 | 30 | 1.444 | 0.625 | 2.311 | 0.2697 | 0.1661 |
| b_9a5a45d6e498 | v7 | 678 | 83 | 1.438 | 1.729 | 0.832 | 0.3415 | 0.3506 |
| b_9a5a45d6e498 | cens | 678 | 83 | 1.477 | 1.729 | 0.854 | 0.3711 | 0.3791 |
| b_a0e6376410cb | v7 | 1056 | 57 | 0.908 | 1.188 | 0.764 | 0.5789 | 0.406 |
| b_a0e6376410cb | cens | 1056 | 57 | 0.834 | 1.188 | 0.702 | 0.6243 | 0.3711 |
| b_ad4a5872f2db | v7 | 644 | 34 | 2.139 | 0.708 | 3.02 | 0.1703 | 0.1763 |
| b_ad4a5872f2db | cens | 644 | 34 | 2.233 | 0.708 | 3.153 | 0.1866 | 0.2044 |
| b_afdd8b1a066d | v7 | 2212 | 16 | 0.194 | 0.333 | 0.581 | 0.7425 | 0.3122 |
| b_afdd8b1a066d | cens | 2212 | 16 | 0.267 | 0.333 | 0.801 | 0.8006 | 0.3333 |
| b_b57e86051e7e | v7 | 63 | 12 | 0.14 | 0.25 | 0.558 | 0.9866 | 0.176 |
| b_b57e86051e7e | cens | 63 | 12 | 0.166 | 0.25 | 0.663 | 0.9866 | 0.176 |
| b_b7e9d5634ad3 | v7 | 122 | 10 | 0.305 | 0.208 | 1.462 | 0.4967 | 0.2083 |
| b_b7e9d5634ad3 | cens | 122 | 10 | 0.255 | 0.208 | 1.224 | 0.4971 | 0.2083 |
| b_b9acdf6071f3 | v7 | 350 | 6 | 0.145 | 0.125 | 1.161 | 0.8734 | 0.125 |
| b_b9acdf6071f3 | cens | 350 | 6 | 0.137 | 0.125 | 1.099 | 0.8 | 0.125 |
| b_be0092b8c92e | v7 | 821 | 18 | 0.238 | 0.375 | 0.635 | 0.338 | 0.097 |
| b_be0092b8c92e | cens | 821 | 18 | 0.2 | 0.375 | 0.532 | 0.3782 | 0.1156 |
| b_c2dc4330feb5 | v7 | 96 | 0 | 0.032 | 0.0 | 1.551 | - | - |
| b_c2dc4330feb5 | cens | 96 | 0 | 0.02 | 0.0 | 0.942 | - | - |
| b_c376fc4d6ccb | v7 | 125 | 6 | 0.122 | 0.125 | 0.979 | 0.1926 | 0.125 |
| b_c376fc4d6ccb | cens | 125 | 6 | 0.105 | 0.125 | 0.839 | 0.1689 | 0.125 |
| b_ce7925d56654 | v7 | 738 | 6 | 0.029 | 0.125 | 0.23 | 0.6385 | 0.125 |
| b_ce7925d56654 | cens | 738 | 6 | 0.028 | 0.125 | 0.224 | 0.8734 | 0.125 |
| b_d45195c31e8f | v7 | 55 | 12 | 0.018 | 0.25 | 0.071 | 0.2106 | 0.2202 |
| b_d45195c31e8f | cens | 55 | 12 | 0.015 | 0.25 | 0.062 | 0.2323 | 0.2202 |
| b_dabc992b5f46 | v7 | 136 | 18 | 0.212 | 0.375 | 0.566 | 0.7152 | 0.2373 |
| b_dabc992b5f46 | cens | 136 | 18 | 0.279 | 0.375 | 0.743 | 0.7658 | 0.2739 |
| b_fcce8f31d315 | v7 | 74 | 12 | 0.164 | 0.25 | 0.657 | 0.9294 | 0.1778 |
| b_fcce8f31d315 | cens | 74 | 12 | 0.162 | 0.25 | 0.648 | 0.9936 | 0.1778 |

## Drift permutation importance (production cens model)

```json
[
  {
    "feature": "horizon",
    "delta_mae": 0.011629
  },
  {
    "feature": "temp_now",
    "delta_mae": 0.003229
  },
  {
    "feature": "season_sin",
    "delta_mae": 0.002438
  },
  {
    "feature": "temp_lifetime",
    "delta_mae": 0.002067
  },
  {
    "feature": "temp_std",
    "delta_mae": 0.001826
  },
  {
    "feature": "age_days",
    "delta_mae": 0.001728
  },
  {
    "feature": "voltage_compensated",
    "delta_mae": 0.001384
  },
  {
    "feature": "observations",
    "delta_mae": 0.00111
  },
  {
    "feature": "season_cos",
    "delta_mae": 0.001108
  },
  {
    "feature": "voltage_max",
    "delta_mae": 0.001102
  },
  {
    "feature": "voltage",
    "delta_mae": 0.00103
  },
  {
    "feature": "temp_outlook_42",
    "delta_mae": 0.001026
  },
  {
    "feature": "beta_30",
    "delta_mae": 0.000791
  },
  {
    "feature": "slope_comp_14",
    "delta_mae": 0.000514
  },
  {
    "feature": "drawdown",
    "delta_mae": 0.000464
  }
]
```

## Feature shift on the worst-transferring holdouts (KS held-out vs train)

### hard_small10

| feature | KS | median held-out | median train | shift / train IQR |
|---|---|---|---|---|
| v_std_30 | 0.293 | 0.02465 | 0.01169 | 0.742 |
| voltage | 0.288 | 2.64821 | 2.79708 | -0.475 |
| voltage_compensated | 0.281 | 2.66472 | 2.79581 | -0.426 |
| voltage_max | 0.24 | 3.07747 | 3.11187 | -0.49 |
| beta_30 | 0.234 | 0.00961 | 0.00683 | 0.392 |
| temp_lifetime | 0.228 | 18.75672 | 20.27336 | -0.424 |
| beta_7 | 0.22 | 0.00988 | 0.00704 | 0.389 |
| temp_std | 0.18 | 2.47715 | 2.41196 | 0.028 |
| temp_now | 0.176 | 18.00417 | 20.0 | -0.333 |
| slope_90 | 0.164 | -0.00085 | -0.00056 | -0.256 |
| slope_30 | 0.15 | -0.00091 | -0.00048 | -0.315 |
| age_days | 0.146 | 719.0 | 670.0 | 0.16 |
| beta_rise | 0.122 | 1.08673 | 1.15961 | -0.176 |
| observations | 0.074 | 601.0 | 593.0 | 0.03 |
| season_cos | 0.072 | 0.30574 | 0.27695 | 0.023 |
| v_std_rise | 0.071 | 1.14704 | 1.14679 | 0.0 |
| staleness | 0.06 | 0.0 | 0.0 | 0.0 |
| season_sin | 0.056 | -0.0387 | 0.04514 | -0.056 |
| days_below_2.50 | 0.047 | -1.0 | -1.0 | 0.0 |
| days_below_2.45 | 0.008 | -1.0 | -1.0 | 0.0 |

### hard_betashift5

| feature | KS | median held-out | median train | shift / train IQR |
|---|---|---|---|---|
| v_std_30 | 0.659 | 0.03806 | 0.01125 | 1.622 |
| beta_30 | 0.654 | 0.01361 | 0.00667 | 1.023 |
| beta_7 | 0.629 | 0.01357 | 0.00683 | 0.952 |
| voltage | 0.558 | 2.56737 | 2.80407 | -0.775 |
| voltage_compensated | 0.539 | 2.59394 | 2.80305 | -0.695 |
| temp_lifetime | 0.38 | 18.3167 | 20.36259 | -0.55 |
| slope_90 | 0.334 | -0.00115 | -0.00054 | -0.541 |
| temp_now | 0.319 | 17.80952 | 20.0 | -0.365 |
| age_days | 0.315 | 765.0 | 662.0 | 0.339 |
| slope_30 | 0.238 | -0.00114 | -0.00047 | -0.498 |
| observations | 0.213 | 659.0 | 589.0 | 0.265 |
| season_cos | 0.203 | 0.58482 | 0.27695 | 0.25 |
| days_below_2.50 | 0.198 | -1.0 | -1.0 | 0.0 |
| voltage_max | 0.189 | 3.1089 | 3.11006 | -0.016 |
| temp_std | 0.17 | 2.58844 | 2.41196 | 0.077 |
| days_below_2.45 | 0.147 | -1.0 | -1.0 | 0.0 |
| v_std_rise | 0.116 | 1.21181 | 1.13948 | 0.127 |
| beta_rise | 0.102 | 1.13141 | 1.15721 | -0.061 |
| season_sin | 0.094 | -0.15845 | 0.04514 | -0.136 |
| staleness | 0.022 | 0.0 | 0.0 | 0.0 |

### hard_mosteol5

| feature | KS | median held-out | median train | shift / train IQR |
|---|---|---|---|---|
| age_days | 0.252 | 738.5 | 661.0 | 0.346 |
| voltage_max | 0.189 | 3.12447 | 3.10584 | 0.307 |
| observations | 0.181 | 535.0 | 606.0 | -0.336 |
| beta_30 | 0.167 | 0.00793 | 0.00634 | 0.234 |
| beta_7 | 0.166 | 0.00818 | 0.00657 | 0.225 |
| voltage_compensated | 0.165 | 2.77056 | 2.79476 | -0.077 |
| voltage | 0.156 | 2.76827 | 2.79917 | -0.095 |
| slope_90 | 0.122 | -0.00072 | -0.00049 | -0.197 |
| staleness | 0.098 | 0.0 | 0.0 | 0.0 |
| slope_30 | 0.091 | -0.00063 | -0.00043 | -0.155 |
| v_std_30 | 0.086 | 0.01258 | 0.01189 | 0.038 |
| temp_std | 0.078 | 2.24144 | 2.50148 | -0.111 |
| temp_lifetime | 0.071 | 20.09622 | 20.33176 | -0.066 |
| v_std_rise | 0.07 | 1.17392 | 1.12715 | 0.081 |
| beta_rise | 0.069 | 1.15787 | 1.15415 | 0.009 |
| temp_now | 0.068 | 20.0 | 20.0 | 0.0 |
| season_sin | 0.046 | -0.0387 | 0.04514 | -0.055 |
| days_below_2.50 | 0.044 | -1.0 | -1.0 | 0.0 |
| days_below_2.45 | 0.032 | -1.0 | -1.0 | 0.0 |
| season_cos | 0.031 | 0.27695 | 0.27695 | 0.0 |

## Beta scale vs beta rise: per-building dispersion

```json
{
  "beta_30": {
    "n_buildings": 24,
    "median_of_medians": 0.00353,
    "iqr_over_median": 0.724,
    "cv": 0.469,
    "max_over_min": 5.79
  },
  "beta_rise": {
    "n_buildings": 24,
    "median_of_medians": 1.07206,
    "iqr_over_median": 0.056,
    "cv": 0.054,
    "max_over_min": 1.32
  },
  "v_std_30": {
    "n_buildings": 24,
    "median_of_medians": 0.00754,
    "iqr_over_median": 1.002,
    "cv": 0.682,
    "max_over_min": 14.63
  },
  "v_std_rise": {
    "n_buildings": 24,
    "median_of_medians": 1.03477,
    "iqr_over_median": 0.065,
    "cv": 0.066,
    "max_over_min": 1.42
  },
  "voltage": {
    "n_buildings": 24,
    "median_of_medians": 2.95323,
    "iqr_over_median": 0.038,
    "cv": 0.031,
    "max_over_min": 1.14
  },
  "margin_scenario_rows": {
    "n_buildings": 24,
    "median_of_medians": 0.28177,
    "iqr_over_median": 0.957,
    "cv": 0.497,
    "max_over_min": 29.67
  },
  "margin_quantile_within_fleet_day": {
    "n_buildings": 24,
    "median_of_medians": 0.2719,
    "iqr_over_median": 1.759,
    "cv": 0.652,
    "max_over_min": 116.46
  }
}
```
