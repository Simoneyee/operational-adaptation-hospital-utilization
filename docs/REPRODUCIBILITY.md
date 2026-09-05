# Reproducibility Guide

## 1. Environment

Recommended:

```bash
conda env create -f environment.yml
conda activate ajph-operational-adaptation
```

or:

```bash
python -m venv .venv
pip install -r requirements.txt
```

The analyses were developed in a Windows / Anaconda / Spyder environment.

## 2. Local path configuration

Historical analysis scripts may contain absolute Windows path defaults from the development environment.

Before execution:

1. clone this repository;
2. download the public source datasets;
3. edit the `ROOT`, raw-data, and clean-data path variables near the top of each script;
4. keep raw data outside version control.

A future refactor may centralize these settings into a configuration file. The archived scripts are retained close to the versions used for the manuscript analyses.

## 3. Core execution order

Run:

```text
analysis/core/01_empirical_upgrade_pipeline.py
analysis/core/02_log_response_state_bridge.py
analysis/core/03_pretrend_adjusted_change.py
analysis/core/04_orthogonal_dml.py
analysis/core/05_mechanism_heterogeneity.py
analysis/core/06_methodological_validation.py
analysis/core/07_denominator_independent.py
analysis/core/08_temporal_preonly_sensitivity.py
```

## 4. Key definitions

### Surge signal

- adult COVID-19 admissions >=5;
- >=50% above preceding 4-week median;
- >=6 weeks between successive hospital episodes.

### Response window

Staffed-capacity adaptation is measured during event weeks +1 and +2.

### Original follow-up

Weeks +2:+4 relative to pre-surge weeks -4:-2.

### Strict-lag sensitivity

Weeks +3:+5 and +3:+6, removing overlap with the +1:+2 response window.

## 5. Statistical inference

Primary empirical inference uses hospital-grouped cross-fitted orthogonal estimation with **state-clustered standard errors**.

The standard DoubleML implementation is an implementation replication, not a replacement for the state-clustered primary inference.

## 6. Interpretation

The empirical design remains observational.

The denominator-independent analyses show that high-response episodes have:

- lower capacity-relative occupancy;
- higher fixed-precapacity occupied burden;
- higher absolute occupied-bed census.

Therefore the main empirical finding should be interpreted as a utilization pattern **consistent with greater absorptive capacity**, not as evidence that adaptation reduced patient burden or caused greater throughput.

## 7. Reproducible release

For a manuscript-linked archival release:

1. verify all scripts;
2. remove local-only files;
3. create GitHub release `v1.0.0`;
4. archive the release in Zenodo;
5. insert the Zenodo DOI into `CITATION.cff`, README, and the manuscript Data Availability statement.
