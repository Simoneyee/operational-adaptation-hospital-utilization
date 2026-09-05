# Variable Notes

## hospital_log_response_innovation

Continuous operational-response index.

Conceptually:

```text
(post observed log-capacity change - post expected log-capacity change)
-
(pre observed log-capacity change - pre expected log-capacity change)
```

Positive values represent greater-than-expected post-surge staffed-capacity adaptation relative to the hospital's immediate pre-surge pattern.

This variable is **not** direct surveillance-to-decision latency.

## delta_occupancy

Change in adult inpatient occupancy ratio from the pre-surge window to the post-surge window.

Because staffed beds are the denominator of occupancy and capacity adaptation contributes to the exposure, denominator-independent outcomes are required for interpretation.

## fixed-precapacity occupied-bed burden

```text
(post occupied beds - pre occupied beds) / pre staffed beds
```

The denominator is held at pre-surge staffed capacity.

## 100 x Delta ln(occupied-bed census)

Proportional occupied-bed census outcome. A value of approximately 4.26 corresponds to roughly a 4.35% proportional increase.

## Pretreatment-only response sensitivity

Expected post response excludes contemporaneous +1/+2 demand change and uses baseline, pre-surge, signal-time, and calendar predictors only.
