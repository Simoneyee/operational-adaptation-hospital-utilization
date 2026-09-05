# Operational Adaptation and Capacity-Relative Hospital Utilization During Epidemic Surges

Analysis code and reproducibility materials supporting the manuscript:

> **Operational Adaptation and Capacity-Relative Hospital Utilization During Epidemic Surges: Mechanistic and National U.S. Evidence**

**Author:** Wenbo Tang  
**Affiliation:** School of Finance and Statistics, Hunan University  
**ORCID:** 0009-0005-9513-1963

## Overview

This repository contains the analysis pipeline for a study combining a fixed mechanistic epidemic-response simulation with national U.S. hospital-level observational analyses.

The empirical component evaluates whether greater-than-expected hospital capacity adaptation after surge signals is associated with a reproducible pattern of:

- lower **capacity-relative occupancy**;
- higher **absolute occupied-bed census**; and
- higher **fixed-precapacity patient burden**.

This pattern is interpreted as **consistent with greater absorptive capacity**, while remaining compatible with demand-driven adaptation. The empirical analyses are observational and are not presented as causal estimates.

## Repository structure

```text
analysis/
  core/
    01_empirical_upgrade_pipeline.py
    02_log_response_state_bridge.py
    03_pretrend_adjusted_change.py
    04_orthogonal_dml.py
    05_mechanism_heterogeneity.py
    06_methodological_validation.py
    07_denominator_independent.py
    08_temporal_preonly_sensitivity.py
  robustness/
    01_objective_quality_state_bridge.py
    02_episode_fe_dynamic.py
  secondary/
    01_hsa_mortality.py
    02_arnorth_external_validation.py
docs/
  DATA_SOURCES.md
  REPRODUCIBILITY.md
  VARIABLE_NOTES.md
outputs/
  derived/
```

## Main analysis sequence

The recommended execution order is:

1. `analysis/core/01_empirical_upgrade_pipeline.py`
2. `analysis/core/02_log_response_state_bridge.py`
3. `analysis/core/03_pretrend_adjusted_change.py`
4. `analysis/core/04_orthogonal_dml.py`
5. `analysis/core/05_mechanism_heterogeneity.py`
6. `analysis/core/06_methodological_validation.py`
7. `analysis/core/07_denominator_independent.py`
8. `analysis/core/08_temporal_preonly_sensitivity.py`

Robustness and secondary-validation scripts can then be run from their respective folders.

## Locked hospital-surge definition

A hospital surge signal was defined by:

- adult COVID-19 admissions **>= 5**;
- admissions **>= 50% above** the hospital's median admissions during the preceding **4 weeks**; and
- a minimum **6-week separation** between successive surge episodes within the same hospital.

## Principal empirical sample

The final orthogonal-analysis panel contains:

- **12,232 hospital-surge episodes**;
- **2,594 hospitals**; and
- the **50 U.S. states plus the District of Columbia**.

## Primary empirical result

In the locked cross-fitted orthogonal model, a 1-SD greater residualized response innovation was associated with an approximately **2.93-percentage-point greater decline in capacity-relative occupancy**.

Denominator-independent analyses showed higher absolute census rather than lower census, changing the substantive interpretation from reduced patient burden to a pattern consistent with greater hospital absorptive capacity.

Strict-lag (+3:+5 and +3:+6) and pretreatment-only response sensitivities preserved the same qualitative pattern.

## Data sources

Raw source data are **not redistributed** here. They are available from the original public providers. See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Reproducibility

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for environment setup, file placement, execution order, and interpretation notes.

## Important analytic caveats

- Hospital response innovation is an operational-response index, not a direct timestamped measure of surveillance-to-decision latency.
- The primary occupancy ratio contains staffed beds in its denominator; denominator-independent analyses are therefore essential to interpretation.
- Post-signal capacity adaptation is endogenous to evolving demand.
- State and substate mortality analyses are secondary and imprecise.
- External military-support analyses failed the prespecified pretrend diagnostic and are retained only for transparency.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). A versioned DOI can be added after archiving a GitHub release in Zenodo.

## License

Code is released under the MIT License. Public source datasets remain subject to their original providers' terms.
