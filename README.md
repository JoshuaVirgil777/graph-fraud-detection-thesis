# Graph-Based Anomaly Detection in Retail Transactions

Code and results for my Master's thesis at Unitec Institute of Technology: **does representing card transactions as a heterogeneous graph improve fraud detection compared to tabular gradient boosting?**

**Short answer:** not on the real dataset. On the feature-sparse simulated dataset, cheap prior-only aggregates beat GNN embeddings — but the gain over plain XGBoost is small.

## Research question

Fraud is rare (0.5–3.5% of transactions) and adversarial. Most production detectors are gradient-boosted trees looking at one row at a time. Graph neural networks (GNNs) can in principle use the fact that the same card, billing address, device and email domain recur — but the literature disagrees on whether they actually do.

This thesis tests the **information-gap hypothesis**: the marginal value of an explicit graph falls as relational information is already encoded in the tabular features. Two datasets are chosen for the contrast:

| Dataset | Type | Size | Fraud | Relational info in tabular features? |
|---|---|---|---|---|
| IEEE-CIS | Real e-commerce | 590,540 | 3.50% | Yes — Vesta's `C1`–`C14` counts are pre-aggregated card/address statistics |
| Sparkov | Simulated | 463,098 (25% sample) | 0.52% | No — raw amount, merchant, category, time, demographics |

Both use a chronological 72/8/20 split (fit / validation / test).

## Headline results

Three findings hold across all seeds and both datasets:

1. **Plain XGBoost beat the standalone GAT by a wide margin** — AUPRC 0.512 vs. 0.163–0.225 on IEEE-CIS; 0.866 vs. 0.325–0.410 on Sparkov.
2. **Neighbour messages helped the standalone GAT on Sparkov (+0.085) and hurt it on IEEE-CIS (−0.062).** The sign depends on what the tabular features already contain.
3. **Removing the IEEE-CIS C-columns cost 0.045 AUPRC, and cheap degree counts did not recover any of it.**

Two results look stronger than they are:

- The **hybrid's gain over plain XGBoost** has a bootstrap interval that includes zero on both datasets (+0.0025 on IEEE-CIS, +0.0105 on Sparkov).
- **SHAP attribution** mostly reflects block size, not added accuracy — an embedding block that's 6.9% of inputs can account for 29.2% of attribution without moving AUPRC.

Every comparison is graded *Supported / Suggestive / Not detected*, requiring both a bootstrap interval excluding zero and all three seed-level deltas agreeing in sign. See Section 4 of the thesis for the full grading rule.

## Repository layout

```
.
├── thesis_code_part1_IEE-CIS.ipynb    # IEEE-CIS pipeline (loading, graph, GAT, trees, SHAP)
├── thesis_code_part2_Sparkov.ipynb    # Sparkov pipeline (same structure, different data)
├── results_v2/                        # IEEE-CIS outputs (created on first run)
├── sparkov_outputs/                   # Sparkov outputs (created on first run)
└── README.md
```

Both notebooks are self-contained and **resumable** — if a run gets interrupted, the next execution picks up from the last completed seed.

## Data

Neither dataset is in this repository (size and licensing).

**IEEE-CIS** — [Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection):
- `train_transaction.csv`
- `train_identity.csv`

Put both in the same folder. The notebook searches your Desktop and home directory; set `THESIS_IEEE_DIR` to override.

**Sparkov** — [Kaggle dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection):
- `fraudTrain.csv`
- `fraudTest.csv`

Place in `~/Downloads` (default) or set `THESIS_SPARKOV_DIR`.

The original Sparkov generator is by [Brandon Harris](https://github.com/namebrandon/Sparkov_Data_Generation); the Kaggle release above is the version used here.

## Running

Developed and tested on an Apple silicon laptop (arm64, 10 cores, 17.2 GB RAM, CPU only). It will run elsewhere; expect different runtimes.

**Smoke test** (synthetic data, tiny models, under a minute):

```bash
THESIS_SMOKE=1 jupyter notebook thesis_code_part1_IEE-CIS.ipynb
```

**IEEE-CIS full run** (~23 minutes with `PROFILE=fast`, longer with `PROFILE=full`):

```bash
THESIS_PROFILE=fast THESIS_RUN_GAT=1 jupyter notebook thesis_code_part1_IEE-CIS.ipynb
```

**Sparkov full run** (~5–8 minutes):

```bash
SPARKOV_FRAC=0.25 SPARKOV_GAT=1 jupyter notebook thesis_code_part2_Sparkov.ipynb
```

Outputs go to `results_v2/` and `sparkov_outputs/`, each with a `run_info.json` or `environment.json` recording the exact versions and settings used.

## Environment

| | |
|---|---|
| Python | 3.9.6 |
| PyTorch | 2.8.0 (CPU) |
| PyTorch Geometric | 2.6.1 |
| XGBoost | 2.1.4 |
| LightGBM | 4.6.0 |
| scikit-learn | 1.6.1 |
| NumPy | 2.0.2 |
| pandas | 2.3.3 |
| SHAP | (latest at time of writing) |

**Import order matters** — torch must be imported before XGBoost and LightGBM to avoid OpenMP thread conflicts on macOS. Both notebooks set the necessary environment variables (`KMP_DUPLICATE_LIB_OK`, `PYTORCH_ENABLE_MPS_FALLBACK`) at the top of Cell 1.

## Caveats

- **The GATs are undertrained.** Every GAT run on both datasets hit the epoch cap with validation AUPRC still rising; the wall-clock caps were never reached. Treat GAT scores as lower bounds. This is the biggest single caveat.
- **The Sparkov LightGBM result is unexplained** (one seed, AUPRC 0.608 vs. 0.866 for XGBoost). Most likely an early-stopping artifact from passing average precision to `fit` alongside the default log-loss. Reported, flagged, and excluded from inference.
- **Embeddings are in-sample** for the fit slice — no out-of-fold scheme. This is a threat to every hybrid result.
- **SHAP rests on 2,000 test rows and one seed** — roughly 70 fraud cases on IEEE-CIS, about 8 on Sparkov. The top few features are trustworthy; the tail is not.
- **Only one GNN architecture** was tested (GAT). No GraphSAGE, no heterophily-aware models.

Section 4.7 of the thesis enumerates every limitation; Section 5.4 lists the follow-up experiments that would sharpen or overturn the conclusions.

## Citation

```bibtex
@mastersthesis{lazar2026graph,
  title  = {Graph-Based Anomaly Detection in Retail Transactions},
  author = {Lazar, Joshua Virgil John},
  school = {Unitec Institute of Technology},
  year   = {2026},
  type   = {Master's thesis},
  note   = {Supervisors: Dr. Eltahir Kabbar and Dr. Bashar Barmada}
}
```

## Author

Joshua Virgil John Lazar, 
Department of Computing, Electrical and Applied Technology
Unitec Institute of Technology
