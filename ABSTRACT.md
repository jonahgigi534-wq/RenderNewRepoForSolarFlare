# Official Abstract (≤250 words) — REFERENCE DRAFT
<!-- The team writes the submitted abstract from scratch in their own words;
     this draft tracks the current verified numbers to draw from. -->

**Benchmark or Reality? Quantifying the Operational Skill Gap in Machine-Learning
Solar-Flare Forecasting**

*Jonathan Gigi, Alfred Antony, Aidan George — Cypress Woods High School*

Machine-learning solar-flare forecasts are evaluated on curated benchmarks and
routinely report 90–99% accuracy. Whether those scores reflect real-time
operational performance on the raw satellite data a deployed system consumes
has not been tested. We ask whether benchmark scores overstate
operational skill and whether live-data training closes the gap.

Using a RandomForest trained on the SWAN-SF benchmark (True Skill Statistic,
TSS, 0.77), we built an out-of-sample evaluation from live JSOC/SDO
magnetic-field data and NOAA flare records across three solar-cycle phases. At
its default operating point the model's TSS fell to 0.35 (2014), 0.64 (2015),
and 0.53 (2017) — below the benchmark everywhere, and below its
cluster-bootstrap 95% confidence interval in 2014 and 2017. A 2×2 experiment
showed that training on live data does not close the gap — the overstatement is
a property of the benchmark's evaluation set. The loss is largely a threshold
artifact: the default threshold optimized F1, which collapses as the flare base
rate falls. Re-tuning for the base-rate-robust TSS — from benchmark validation
data alone — recovered TSS to 0.82–0.84 on unseen years (paired gain +0.19 on
2015, CI excluding zero); recalibrating on live data added at most +0.02.
Auditing our own evaluation exposed catalog decay — flare attribution fell from
94–98% (2013–2017) to 15% (2023) — so a fail-closed label-quality gate now
guards every evaluation. A zero-skill model matches "99% accuracy" — accuracy
is meaningless here.

These results establish a reproducible method for honestly evaluating
operational flare forecasts and caution that benchmark scores must not be read
as real-world performance.
