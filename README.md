# Hybrid wear-rate model for monotone tool wear estimation (code and data)

Code and derived data for the paper:

> D. Katamba Mpoyi, C. Bagalwa Bienvenu, A. A. Nanha Djanan, K. Y'Ebondo Bovic, G. Pappalettera and C. Casavola, *Monotone estimation of milling tool wear with a hybrid wear-rate model and conformal prediction intervals*, submitted to Measurement Science and Technology.

The repository reproduces every number, table and figure of the paper (except figure 2, a drawing of the workflow) from the public NASA Ames milling data set.

## Method in brief

For each pass *k* of an insert, an empirical wear law fitted per material (power law, or three-stage law) gives the expected wear increment ΔVB<sub>k</sub><sup>phys</sup> ≥ 0. A small neural network fed with spindle-current, vibration and acoustic-emission descriptors, the cutting parameters and the previous estimate returns *h<sub>k</sub>*, and

VB̂<sub>k</sub> = VB̂<sub>k−1</sub> + ΔVB<sub>k</sub><sup>phys</sup> · exp(*h<sub>k</sub>*),  VB̂<sub>0</sub> = 0,

so the estimate cannot decrease. Split conformal prediction, calibrated on inserts not used for training, adds a 90 % interval used for the tool-change decision. Models are evaluated by leave-one-case-out and leave-one-condition-out cross-validation with nested model selection. The implementation is in `src/wearlib.py` (`HybridWearRate`, `PowerLaw`, `ThreeStageLaw`, `fit_predict`).

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
sh run_all.sh
```

`run_all.sh` downloads the NASA archive (checksum verified), rebuilds the features, runs all experiments, writes the tables and figures, and opens the figures of the paper on screen (add `--no-show` to `src/04_make_figures.py` to skip this, e.g. on a server). It takes about 30 minutes on a 10-core laptop, CPU only. To check the analysis without re-training, skip the first three steps: the outputs of the experiments are included in `results/raw/`, so `python3 src/03_analyse.py` and `python3 src/04_make_figures.py` run in a few seconds and show the figures directly.

## Repository layout

| Path | Content |
|---|---|
| `data/download_data.py` | downloads and checks the NASA archive into `data/raw/` |
| `data/processed/` | per-pass descriptors (`features.csv`) and per-case summary; see `data/README.md` |
| `src/wearlib.py` | models (wear laws, HWR, references), training, metrics |
| `src/01_prepare_data.py` | screening and descriptor extraction |
| `src/02_run_experiments.py` | nested cross-validation of all models, 5 seeds |
| `src/03_analyse.py` | metrics, paired tests, conformal intervals, tool-change decision |
| `src/04_make_figures.py` | figures of the paper and supplementary figures |
| `results/raw/` | predictions of every model, seed and fold; inner out-of-fold predictions; selected settings; intervals |
| `results/tables/` | all tables as CSV |
| `results/figures/` | figures as PDF and PNG |

Model names in the code and CSV files: `EMP` power law alone, `EMP3` three-stage law alone, `RF` random forest, `MLP`, `GRU`, `PINN-soft` soft-constraint physics-informed network, `HWR` hybrid model with power-law prior (HWR-P in the paper), `HWR3` with three-stage prior (HWR-T), `HWR-sel` prior selected in each fold (**HWR, the proposed model**), and `HWR-noprior`, `HWR-nofeedback`, `HWR-labelled-only` for the ablations. Protocols: `LOCO` leave-one-case-out, `LOCD` leave-one-condition-out. Seed `ens` is the five-seed ensemble.

## Where each result of the paper comes from

| Paper | File |
|---|---|
| Table 1 | `data/processed/dataset_summary.csv` |
| Table 2 | `results/tables/T2_overall_metrics.csv` (columns `RMSE_ens`, `MAE_ens`, `R2_ens`) |
| Table 3 | `results/tables/T6_ablation.csv` |
| Table 4 | `results/tables/T7_conformal_intervals.csv` |
| Table 5 | `results/tables/T8_tool_change_summary.csv` (protocol `LOCO`) |
| Paired tests and p-values in the text | `results/tables/T5_paired_tests.csv` |
| Per-case and per-material errors in the text | `results/tables/T3_per_case_RMSE_*.csv`, `T4_per_material.csv` |
| Prior chosen in each fold | `results/tables/T9b_prior_selected_per_fold.csv` |
| Figure 1 | `results/figures/fig1_dataset.pdf` |
| Figure 3 | `results/figures/fig3_trajectories.pdf` |
| Figure 4 | `results/figures/fig4_monotonicity.pdf` |
| Figure 5 | `results/figures/fig5_multiplier.pdf` |
| Other figures | `results/figures/supp_*.pdf` (supplementary, not in the paper) |

## Reproducibility notes

- Random seeds are fixed (0–4 for each model). The results were produced with the versions in `requirements.txt` on macOS 15 (Apple arm64, CPU). Other platforms or library versions may give small differences in the neural-network results, not in the empirical laws.
- Hyper-parameter grids are in `wearlib.GRIDS`. They were fixed after a preliminary exploration under leave-one-case-out; the selection reported in the paper is nested.
- The soft-constraint network is our re-implementation of the penalty idea of Fang *et al* (2025) and Chen *et al* (2026) for data without cutting forces, not the authors' code.

## Licence and citation

Code under the MIT licence (see `LICENSE`). The NASA data set is not covered by this licence; please cite it as indicated in `data/README.md`. If you use this code, please cite the paper (see `CITATION.cff`).
