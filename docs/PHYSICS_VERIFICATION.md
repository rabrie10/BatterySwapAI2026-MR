# Data-physics verification report

_Generated 2026-08-22 03:44 from dataset/train. Measurement only; no model fitted._
_Smoothing: exact reimplementation of official smooth_series (7-day trailing median of daily medians, 10<T<30 filter, >=5 readings/day, min_periods=int(0.5*7)=3 as pinned by tests/test_smoothing.py)._

## Key flags (hypothesis check)

1. **Monotone-state hypothesis only PARTIALLY supported.** Temperature compensation removes some but NOT most of the upward movement: up/down mass ratio goes 0.575 (raw) -> 0.464 (global comp) -> 0.453 (per-device comp, best variant). At a 28-day lag it is 0.321 -> 0.232: a quarter of the monthly-scale movement is still upward after removing the linear temperature term. NOTE: the task-specified whole-life per-device beta (variant c) makes monotonicity WORSE (0.669) because the whole-life regression is confounded by trend x season; the detrended beta is the physical one.
2. **Shared-curve x per-device-rate model CONTRADICTED.** Band first-passage times have CV 0.77-0.90 and boundary-free cross-band correlations of log times are ~0 (t(2.8->2.7) vs t(2.6->2.5): r=0.04; t(2.9->2.8) vs t(2.5->2.4): r=-0.03). A device slow in one band is NOT slow in the others. Yet total 2.8->2.4 time is far tighter (CV 0.35) than any single band -- plateau dwell and knee speed trade off rather than scale together.
3. **Temperature compensation does NOT tighten the level-threshold warning window** (IQR improves for only 2/5 global, 1/5 per-device thresholds). Warning-time spread is dominated by knee-shape heterogeneity, not temperature: last-day-above-2.55 gives median 65 d warning with IQR 90 d.
4. **The rising-dV/dT (internal-resistance) part of the hypothesis is CONFIRMED.** Within-day beta roughly doubles from its pre-180 d baseline (median 0.00526) well before EOL -- median first 2x exceedance 169 d before crossing (31 of 80 censored at >=180 d), and AUC vs matched surviving device-days at 42 d lead = 0.858. Final-90 d beta is 1.75x early-life beta for EOL devices (94% increase).
5. **Plateau-then-knee shape CONFIRMED**: smoothed EOL trajectories fall only ~152 mV over days -180..-42 but ~112 mV over the last 42 d; final-14 d slope median -0.0042 V/d. But the knee is NOT sharp at 42 d: only 33% are still above 2.55 V then.

## 1. Inventory

- Devices: **461** across **24** buildings (79 rooms); devices/building median 9.0 (min 2, max 117).
- Metrics: 8,520,098 hourly rows, 2022-11-22 08:00:00 to 2026-07-31 07:00:00 (1347 days).
- Non-null EOL times: **82** (first 2025-03-06, last 2026-07-28). Per building: {'b_15bc00efc223': 1, 'b_2b69c4007582': 6, 'b_4217a3b6e58b': 1, 'b_53df020ad893': 5, 'b_55038c6689a0': 1, 'b_60c6d1796ab4': 1, 'b_62e09ab003f8': 7, 'b_6869b0f15a78': 1, 'b_7aac45259c8a': 1, 'b_955300d5f090': 5, 'b_9a5a45d6e498': 15, 'b_a0e6376410cb': 11, 'b_ad4a5872f2db': 6, 'b_afdd8b1a066d': 3, 'b_b57e86051e7e': 2, 'b_b7e9d5634ad3': 2, 'b_b9acdf6071f3': 1, 'b_be0092b8c92e': 3, 'b_c2dc4330feb5': 1, 'b_c376fc4d6ccb': 1, 'b_ce7925d56654': 1, 'b_d45195c31e8f': 2, 'b_dabc992b5f46': 3, 'b_fcce8f31d315': 2}
- First reading minus devices.start_time: median -0.01 d, p90 1.92 d; within 1 day for 88.5% of devices -> start_time is the install/first-observation date (a few stragglers: max gap 922 d).
- First raw voltage: median 3.149 V (IQR 3.111-3.181). First smoothed voltage: median 3.107 V. Bands: >3.05: 74.6%, 2.95-3.05: 20.0%, 2.85-2.95: 3.7%, 2.70-2.85: 1.5%, <=2.70: 0.2%

## 2. Trajectory shape (82 EOL devices, smoothed series)

- Grid crossing (first smooth_v < 2.4) matches eol_times.csv exactly for 100.0% of 82 devices (diff median 0.0 d, max |diff| 0 d).

| days before crossing | median smooth_v | IQR (p25-p75) | n |
|---|---|---|---|
| 0 (at crossing) | 2.395 | 2.385-2.397 | 82 |
| 7 | 2.433 | 2.419-2.453 | 68 |
| 14 | 2.448 | 2.431-2.485 | 72 |
| 28 | 2.479 | 2.453-2.524 | 71 |
| 42 | 2.512 | 2.472-2.565 | 70 |
| 60 | 2.551 | 2.496-2.612 | 68 |
| 90 | 2.591 | 2.522-2.679 | 72 |
| 180 | 2.663 | 2.599-2.760 | 73 |

- **Median smooth_v 42 d before crossing: 2.512 V**; **32.9%** of EOL devices still above 2.55 V at 42 d out (n=70).
- Post-knee slope (OLS over final 14 d): median **-0.0042 V/day** (IQR -0.0068 to -0.0024); 14-day difference quotient median -0.0040 V/day.
- Observed smoothed life before crossing: median 800 d (min 330, p10 549); rows with n<82 above are devices whose observation started less than k days before their crossing.

## 3. Warning window of a pure level threshold

Days between the last day smooth_v > X and the 2.4 V crossing (per EOL device). comp = global 0.00463 V/degC; pd-comp = per-device detrended beta.

| X | raw median | raw IQR | raw CV | comp median | comp IQR | comp CV | pd-comp median | pd-comp IQR | pd-comp CV |
|---|---|---|---|---|---|---|---|---|---|
| 2.70 | 213.0 | 216.5 | 0.64 | 243.0 | 206.2 | 0.60 | 224.0 | 174.8 | 0.59 |
| 2.60 | 95.5 | 103.8 | 0.79 | 101.5 | 117.0 | 0.78 | 104.5 | 118.2 | 0.78 |
| 2.55 | 65.0 | 90.2 | 0.80 | 67.5 | 88.8 | 0.98 | 65.0 | 92.5 | 1.00 |
| 2.50 | 35.0 | 51.5 | 1.00 | 34.0 | 58.5 | 1.02 | 34.0 | 62.8 | 1.09 |
| 2.45 | 16.5 | 21.5 | 1.49 | 13.0 | 25.8 | 1.60 | 13.0 | 34.8 | 1.66 |

- Global compensation tightens the IQR for **2/5** thresholds; per-device compensation for **1/5**.
- Compensated series' own 2.4 V crossing vs official: median shift 0.5 d (IQR 10.8 d, n=74).

## 4. Monotonicity of the smoothed series

Devices with >=180 smoothed days: **455**. Adjacent-day deltas pooled.

| series | frac deltas > 0 | frac >0 (excl. ties) | up/down mass ratio | ratio (only deltas >5 mV) | n deltas |
|---|---|---|---|---|---|
| (a) raw smooth_v | **0.2327** | 0.3603 | **0.5745** | 0.6023 | 333,560 |
| (b) global comp (0.00463) | **0.2659** | 0.3684 | **0.5466** | 0.5396 | 333,560 |
| (c) per-device whole-life-beta comp | **0.2762** | 0.3826 | **0.6686** | 0.7064 | 333,560 |
| (b2) global comp, re-smoothed daily | **0.1995** | 0.3041 | **0.4642** | 0.5259 | 333,560 |
| (c2) per-device whole-life comp, re-smoothed | **0.2199** | 0.3459 | **0.6080** | 0.6676 | 333,560 |
| (d) per-device detrended-beta comp | **0.2664** | 0.3691 | **0.5523** | 0.5470 | 333,560 |
| (d2) per-device detrended comp, re-smoothed | **0.1995** | 0.3037 | **0.4528** | 0.5129 | 333,560 |

28-day-lag deltas (does upward movement survive a monthly horizon?):

| series (lag 28) | frac deltas > 0 | up/down mass ratio |
|---|---|---|
| (a) raw smooth_v | **0.2942** | **0.3213** |
| (b2) global comp, re-smoothed | **0.2225** | **0.2400** |
| (d2) per-device detrended comp, re-smoothed | **0.2274** | **0.2319** |

- Whole-life per-device beta: median 0.00658 V/degC (IQR 0.00277-0.01110); detrended per-device beta: median 0.00511 (IQR 0.00400-0.00656). The whole-life regression is inflated by the aging-trend x season confound, which is why variant (c) can over-correct.

## 5. Shared curve / rate constancy (EOL devices starting > 2.85 V, compensated)

- Devices qualifying (first compensated smooth_v > 2.85): **77** of 82 EOL devices.

| band | median days | IQR | CV | n |
|---|---|---|---|---|
| 2.9->2.8 | 103.5 | 24.8-204.8 | **0.90** | 76 |
| 2.8->2.7 | 111.0 | 55.0-229.0 | **0.77** | 77 |
| 2.7->2.6 | 74.0 | 37.0-178.0 | **0.89** | 77 |
| 2.6->2.5 | 50.0 | 33.0-92.0 | **0.80** | 77 |
| 2.5->2.4 | 51.0 | 28.0-103.0 | **0.87** | 69 |

- Mean pairwise correlation of log band-times: **-0.073** (adjacent bands -0.153, non-adjacent -0.019).
- Halves test (shares fp(2.6) boundary), log t(2.8->2.6) vs log t(2.6->2.4), n=68: Pearson -0.241, Spearman -0.240.
- Boundary-free t(2.8->2.7) vs t(2.6->2.5) (n=67): Pearson **0.038**, Spearman -0.017.
- Boundary-free t(2.9->2.8) vs t(2.5->2.4) (n=60): Pearson **-0.030**, Spearman -0.080.
- Boundary-free t(2.8->2.7) vs t(2.5->2.4) (n=61): Pearson **-0.220**, Spearman -0.181.
- Total 2.8->2.4 time: median 383.0 d (IQR 304.0-462.0, CV 0.35).
- Additive model log(t) = band + device explains R^2 = 0.242; device effect after removing band means: R^2 = 0.161.

## 6. IR signal (within-day dV/dT beta) lead time

| lead (days before crossing) | median trailing-7d beta | IQR | n |
|---|---|---|---|
| 0 | 0.01194 | 0.00947-0.01513 | 82 |
| 14 | 0.01269 | 0.01081-0.01556 | 78 |
| 28 | 0.01214 | 0.00996-0.01451 | 78 |
| 42 | 0.01240 | 0.01000-0.01419 | 79 |
| 60 | 0.01246 | 0.00962-0.01364 | 78 |
| 90 | 0.01133 | 0.00878-0.01331 | 78 |
| 120 | 0.01166 | 0.00855-0.01328 | 76 |
| 180 | 0.01062 | 0.00732-0.01266 | 75 |

- Own baseline (median daily beta, days < crossing-180): median 0.00526 V/degC (n=82).
- Lead time at which trailing-7d beta first exceeds 2x own baseline: median **169.0 d** (IQR 118.5-180.0, n=80; 2 never exceed within 180 d, 0 lacked a baseline; **31** already exceeded at the 180 d scan edge, so their true lead is >=180 d).
- 42-day-lead separation, EOL (n=79, median 0.01240) vs matched surviving device-days (n=23572, median 0.00462): **AUC = 0.858**.

## 7. Temperature structure

- Detrended (60 d rolling-median) daily V vs T: per-device correlation median **0.569** (IQR 0.419-0.687, n=456).
- Detrended per-device beta: median **0.00512 V/degC** (IQR 0.00400-0.00658).
- EOL devices, beta in final 90 d: median 0.01169 vs earlier life 0.00604; paired ratio median 1.75x, final>early for 94.2% of devices.
- Within-day temperature range (unfiltered): median 5.49 degC (IQR 2.00-9.02, p90 13.00).
- Annual swing of smoothed T per building (max minus min monthly mean): median 5.77 degC (min 2.18, max 9.61 across 24 buildings).

## 8. Censoring geometry

- 48 scenarios, planning windows all [42] days, starts 2025-09-01 to 2026-07-27, unobserved_eol_days setting = [30.0].
- Devices whose observation ends inside a 42 d window: median **0.5** per scenario (min 0, max 445; the max comes from the 6 windows that extend past the dataset end and sweep up all administratively-censored devices. Windows fully inside the data: median 0.0, max 14).
- Recorded EOL crossings inside a window: median 9.0 per scenario (min 2, max 18).
- 445 devices are observed to the dataset end; **16** end earlier. Of those early enders, 15 have NO recorded EOL (93.8%) and 100.0% of them were last seen above 2.4 V (median last smooth_v 2.872 V).
- Devices with no EOL overall: last smooth_v median 2.697 V; 100.0% end above 2.4 V.
- After the recorded EOL date, observation continues for median 204.4 d (IQR 112.6-274.4).
- Unobserved-EOL removals inside a window (end of observation with no EOL by window close): median 0.0 per scenario (max 364; windows fully inside the data: median 0.0, max 14).
