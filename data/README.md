# Data

## Source

All results use the public **milling data set of Agogino and Goebel** (BEST Lab, University of California, Berkeley), distributed by NASA Ames Research Center through the NASA Prognostics Data Repository:

- page: https://data.nasa.gov/dataset/milling-wear
- archive: https://data.nasa.gov/docs/legacy/mill.zip (about 15 MB, contains `mill.mat` and `Readme.pdf`)

Please cite it as:
Agogino A and Goebel K 2007 *Milling data set*, BEST Lab, UC Berkeley, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

The archive is **not redistributed** in this repository. `python3 data/download_data.py` downloads it into `data/raw/` and checks it against the SHA-256 checksums of the files used in the paper:

| File | SHA-256 |
|---|---|
| `mill.zip` | `de9c8685cb0e07b4f39459dc282b49192a3ad660e9577ff99b68dc9994e35d27` |
| `mill.mat` | `71486a857939d1416c86c7cf0c469d5e69c7c30495e01ec8a0e13aafd2a313cb` |

## Experiments (summary of the original documentation)

Matsuura MC-510V machining centre, 70 mm face mill with six KC710 inserts, cutting speed 200 m/min (826 rev/min). Depth of cut 0.75 or 1.5 mm, feed 0.25 or 0.5 mm/rev, workpiece cast iron (material 1) or stainless steel J45 (material 2): eight conditions, each run twice with new inserts (16 cases, 167 passes). Flank wear VB was measured with a microscope after some passes. Six signals per pass at 250 Hz: AC and DC spindle motor current, table and spindle vibration, table and spindle acoustic emission (vibration and acoustic emission after an analogue RMS stage, time constant 8 ms).

## Screening applied

| Item | Decision |
|---|---|
| Case 2, pass 1 | removed: values of the order of 1e29 in all six channels |
| Case 12, pass 1 | removed: 15 360 samples instead of 9000, invalid amplitude range |
| Case 6 | kept in `features.csv`, excluded from evaluation (single pass) |
| Result | 165 passes, 145 with VB, 20 without; 15 cases evaluated |

## Processed files (`data/processed/`)

`features.csv` is produced by `src/01_prepare_data.py`: one row per kept pass, sorted by case and time. Descriptors are computed on the steady part of the pass, located on the smoothed DC current (longest interval above 80 % of the 5th–95th percentile range, trimmed by 10 % at each end).

| Column | Unit | Meaning |
|---|---|---|
| `case`, `run` | – | case number (1–16) and pass counter in the original archive |
| `time_min` | min | cutting time at the end of the pass |
| `VB_mm` | mm | measured flank wear (empty when not measured) |
| `DOC_mm`, `feed_mm_rev` | mm, mm/rev | depth of cut and feed |
| `material` | – | 1 = cast iron, 2 = steel |
| `AC_rms`, `AC_p2p`, `AC_kurt` | V, V, – | RMS, 0.1–99.9 % peak-to-peak and excess kurtosis of the AC spindle current |
| `DC_mean`, `DC_std` | V | mean and standard deviation of the DC spindle current |
| `DC_satfrac` | – | fraction of samples at the +10 V acquisition limit |
| `vib_table_*`, `vib_spindle_*`, `AE_table_*`, `AE_spindle_*` | V | mean, standard deviation and 95th percentile of each envelope |
| `steady_len_s` | s | duration of the steady part |

The models use eight of these descriptors (`AC_rms`, `DC_mean`, the four envelope means, `AC_kurt`, `DC_satfrac`), plus their change since the first pass of the same insert, computed in `src/wearlib.py`. The other columns are kept for completeness.

`dataset_summary.csv` gives one row per case (Table 1 of the paper).
