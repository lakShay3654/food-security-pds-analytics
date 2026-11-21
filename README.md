# Is Food-Security Coverage Matched to Need?

**ML Application Project · Problem No. 43 · B.Sc. (H) Computer Science, Semester V**
**Ramanujan College, University of Delhi · Machine Learning (CSA)**

---

## Overview

This project asks whether the statutory food-security coverage shares fixed by the **National Food Security Act (NFSA), 2013** are matched to present-day socioeconomic need across India's 36 States and Union Territories. The shares were originally set from 2011–12 consumption surveys. We ask how well current need indicators (multidimensional poverty and child underweight) predict those shares, which States deviate most, and whether States fall into interpretable need tiers.

| Finding | Value |
|---|---|
| Leave-one-out R² (MPI only) | **0.559** |
| Leave-one-out RMSE (MPI only) | **10.0 pp** |
| Mean-only baseline LOO R² | −0.058 |
| MPI slope (poverty-only OLS) | 0.92 pp coverage per MPI point (95% CI 0.66–1.18) |
| Child underweight adds (nested F-test p) | 0.118 → adds nothing |
| PHH LOO R² | 0.591 |
| AAY LOO R² | 0.089 |
| Flagged units (total coverage, \|z\| > 2) | Andaman & Nicobar (z = −4.1), Manipur (z = +2.9) |
| Bihar residual z-score | −0.99 → within normal scatter |
| K-Means silhouette (K = 3) | 0.436 (weak structure) |

---

## Repository Structure

```
├── Food_829a36.ipynb          ← Main notebook (run top to bottom in Colab)
├── RS_Session_263_AU_68_2.csv ← Coverage table: Rajya Sabha Session 263, Unstarred Q. 68 (OGD India)
├── need_indicators.csv        ← State-level MPI and child underweight with provenance flags
├── README.md
├── requirements.txt
└── outputs/                   ← Created when the notebook runs
    ├── report.md
    ├── results.json
    ├── figures/
    │   ├── fig_pipeline.png
    │   ├── fig1_distributions.png
    │   ├── fig2_correlation.png
    │   ├── fig3_scatter_flagged.png
    │   ├── fig4_diagnostics.png
    │   ├── fig5_residual_zscores.png
    │   ├── fig6_clusters.png
    │   └── fig7_k_selection.png
    └── tables/
        ├── descriptives.csv
        ├── validation_loo.csv
        ├── ols_coefficients.csv
        ├── bootstrap_ols_vs_ridge.csv
        ├── tier_outcomes_validation.csv
        ├── tier_outcomes_flagged.csv
        ├── cluster_profile.csv
        ├── cluster_membership.csv
        ├── residuals_by_state.csv
        └── state_level_results.csv
```

---

## Data Sources

| Dataset | Variable | Source | Reference period |
|---|---|---|---|
| `RS_Session_263_AU_68_2.csv` | Statutory coverage (%); AAY and PHH persons and families | Rajya Sabha, Session 263, Unstarred Q. 68 (answered 2 Feb 2024), [OGD India](https://data.gov.in) | As on 30-01-2024; population from Census 2011 |
| `need_indicators.csv` (MPI column) | Multidimensional poverty headcount ratio (%) | [NITI Aayog National MPI Baseline Report, 2021](https://niti.gov.in/sites/default/files/2021-11/National_MPI_India-11242021.pdf) | NFHS-4, 2015–16 |
| `need_indicators.csv` (underweight column) | Children under 5 underweight, weight-for-age (%) | NFHS-5 State/UT fact sheets; tabulated in [Economic Survey 2021–22, Table 8.17](https://www.indiabudget.gov.in/budget2022-23/economicsurvey/doc/stat/tab817.pdf) | NFHS-5, 2019–21 |

The coverage CSV is not redistributed here if it exceeds OGD's redistribution terms; download it from the OGD link above and place it in the root folder.

### Data provenance flags in `need_indicators.csv`

Each row has `mpi_status` and `malnutrition_status` columns:

- `verified` — checked against the published primary table
- `derived` — computed from published values (see `adjustment_note`)
- `unverified` — transcribed but not yet cross-checked (**action required before submitting**)

**Currently unverified MPI values (20 States):** Andhra Pradesh, Arunachal Pradesh, Assam, Chhattisgarh, Gujarat, Haryana, Himachal Pradesh, Karnataka, Madhya Pradesh, Maharashtra, Manipur, Meghalaya, Mizoram, Nagaland, Odisha, Rajasthan, Telangana, Tripura, Uttarakhand, West Bengal.

**Currently unverified underweight value:** Ladakh (9.1% — confirm in the NFHS-5 Ladakh fact sheet).

---

## How to Run

### Google Colab (recommended)

1. Upload `Food_829a36.ipynb`, `RS_Session_263_AU_68_2.csv` and `need_indicators.csv` to Colab (or place the two CSVs under `/content/`).
2. Open the notebook and click **Runtime → Run all**.
3. The final cells write `outputs/results.json`, `outputs/report.md` and all figures and tables. Download `outputs.zip` from the last cell.

### Locally

```bash
pip install -r requirements.txt
# place the two CSVs in a data/ subfolder or the working directory
jupyter nbconvert --to notebook --execute Food_829a36.ipynb \
    --output Food_829a36_executed.ipynb \
    --ExecutePreprocessor.timeout=900
```

**Random seed:** 42 (set once at the top of the notebook and used throughout).

---

## Methods

| Step | Detail |
|---|---|
| **Outcomes** | Total coverage (%); PHH per 100 pop.; AAY per 100 pop.; AAY share of covered (%) |
| **Predictors** | MPI headcount ratio; child underweight (%); log population (robustness) |
| **Models** | OLS (poverty-only and two-predictor); Ridge (α = 1.0 fixed and α tuned by LOO-CV over 60 values in [10⁻³, 10³]); mean-only baseline |
| **Validation** | Leave-one-out cross-validation; scaling and α refitted inside every fold |
| **Inference** | Standard errors, 95% CI, HC3 robust p-values, nested F-test, VIF |
| **Diagnostics** | Studentised residuals (z-scores), Cook's distance, Breusch–Pagan, Shapiro–Wilk, 2 000-resample bootstrap |
| **Clustering** | K-Means on need features only (coverage excluded); K = 2–6 examined; silhouette, elbow, 30-seed and 200-resample stability; Kruskal–Wallis test across tiers |

---

## Key Results

### Model comparison (leave-one-out)

| Model | LOO R² | LOO RMSE (pp) |
|---|---|---|
| Mean-only baseline | −0.058 | 15.5 |
| OLS: poverty only | **0.559** | **10.0** |
| OLS: poverty + underweight | 0.556 | 10.0 |
| Ridge (α = 1.0) | 0.559 | 10.0 |
| Ridge (α tuned by LOO-CV, α = 1.80) | 0.559 | 10.0 |

Child underweight and ridge shrinkage add nothing measurable. PHH coverage tracks poverty closely (LOO R² = 0.591); AAY coverage barely does (0.089).

### Flagged units (total coverage, |studentised residual| > 2)

| State/UT | Actual | Predicted | Residual (pp) | z-score | Cook's D |
|---|---|---|---|---|---|
| Andaman & Nicobar | 16.65% | 47.68% | −31.0 | −4.1 | 0.26 |
| Manipur | 87.74% | 64.39% | +23.4 | +2.9 | 0.31 |

Bihar (MPI 51.91%, residual −8.2 pp, z = −0.99) is **within normal scatter**.

### Need tiers (K-Means, need features only)

| Tier | N | Mean MPI (%) | Mean coverage (%) | Mean PHH per 100 |
|---|---|---|---|---|
| Low need | 18 | 8.35 | 52.0 | 42.3 |
| Moderate need | 9 | 17.21 | 63.3 | 53.2 |
| High need | 9 | 35.86 | 76.7 | 67.9 |

Kruskal–Wallis p < 0.001. Silhouette 0.436 at K = 3 (weak structure; silhouette peaks at K = 2 at 0.498). Stability: seed ARI mean 0.71, min 0.35.

---

## Limitations

- N = 36 aggregate units; findings are ecological and cannot be transferred to households.
- Three data sources span different reference years (2011–12, 2015–16, 2019–21, 2024).
- MPI includes nutrition, so overlap with child underweight is partly built in.
- 20 MPI values and Ladakh's underweight are unverified (see provenance flags above).
- AAY person counts may be administrative conversions from families (Bihar: exactly 5.00 persons per family).

---

## Citation

```
Lakshay (2026). Is Food-Security Coverage Matched to Need? Evidence from NFSA AAY and PHH Allocations
Across Indian States and Union Territories. ML Application Project, Problem No. 43,
B.Sc. (H) Computer Science Semester V, Ramanujan College, University of Delhi.
```

---

## References (key)

- NITI Aayog. *India National Multidimensional Poverty Index: Baseline Report*, 2021.
- IIPS and ICF. *National Family Health Survey (NFHS-5), 2019–21: State/UT Fact Sheets*, 2021.
- Ministry of Finance. *Economic Survey 2021–22, Statistical Appendix, Table 8.17*, 2022.
- Hoerl & Kennard. Ridge regression. *Technometrics*, 12(1), 1970.
- Pedregosa et al. Scikit-learn. *JMLR*, 12, 2011.
- Rousseeuw. Silhouettes. *J. Comput. Appl. Math.*, 20, 1987.
- Drèze & Sen. *An Uncertain Glory*, Princeton UP, 2013.

Full reference list is in the accompanying project report.

---

*Random seed 42 · Python 3 · scikit-learn · statsmodels · pandas · NumPy · SciPy · seaborn · Matplotlib*
