# Official Abstract (≤250 words)

**Benchmark or Reality? Quantifying the Operational Skill Gap in Machine-Learning
Solar-Flare Forecasting**

*Jonathan Gigi, Alfred Antony, Aidan George — Cypress Woods High School*

Machine-learning solar-flare forecasts are evaluated on curated benchmarks
and routinely report high skill, often 90–99% accuracy. Whether those scores
reflect real-time operational performance — on the raw satellite data a deployed
system consumes — has not been systematically tested. We ask whether
benchmark scores overstate operational skill and whether live-data training
closes the gap.

Using a RandomForest trained on the SWAN-SF benchmark (True Skill
Statistic, TSS, 0.77), we built an out-of-sample evaluation from live
JSOC/SDO magnetic-field data and NOAA flare records, across three solar-cycle
phases. At its default operating point the model's TSS fell to 0.35 (2014),
0.64 (2015), and 0.33 (2023) — below the benchmark everywhere, and below its
cluster-bootstrap 95% confidence interval in two of three periods. A 2×2
experiment showed that training on live data does not close the gap — the
overstatement is a property of the benchmark's evaluation set. The loss is
largely a threshold artifact: the default threshold optimized F1, which
collapses as the flare base rate swings 4.3%→0.8%. Re-tuning for the
base-rate-robust TSS — from benchmark validation data alone — recovered TSS to
0.66–0.84 on unseen years (paired gains +0.19/+0.33, CIs excluding zero);
recalibrating on live data added nothing. All 17 magnetic features shift
between benchmark and operation (median KS D 0.19); the model leans hardest on
the least-stable ones. A zero-skill model matches "99% accuracy" — accuracy is
meaningless here.

These results establish a reproducible method for honestly evaluating
operational flare forecasts and caution that benchmark scores must not be read
as real-world performance.
