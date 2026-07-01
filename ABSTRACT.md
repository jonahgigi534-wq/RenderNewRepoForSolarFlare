# Official Abstract (≤250 words)

**Benchmark or Reality? Quantifying the Operational Skill Gap in Machine-Learning
Solar-Flare Forecasting**

Machine-learning solar-flare forecasts are evaluated on curated benchmark datasets
and routinely report high skill, often 90–99% accuracy. Whether those scores
reflect real-time operational performance — on the raw satellite data a deployed
system actually consumes — has not been systematically tested. We ask whether
benchmark scores overstate operational skill, and whether training on live data
closes the gap.

Using a RandomForest trained on the SWAN-SF benchmark (reported True Skill
Statistic, TSS, 0.77), we built an honest out-of-sample evaluation directly from
live JSOC/SDO magnetic-field data and NOAA flare records, across three periods
spanning different solar-cycle phases. At its default operating point the model's
TSS fell to 0.35 (2014), 0.64 (2015), and 0.33 (2023); the benchmark exceeded the
upper 95% confidence interval of live skill in every period, overstating
operational skill by up to twofold. A controlled 2×2 experiment — benchmark- versus
live-trained models, benchmark versus operational test sets — showed that training
on live data does not close the gap, indicating the overstatement is a property of
the benchmark's evaluation set, not the training data. Kolmogorov–Smirnov tests
found all 17 magnetic input features significantly shifted between benchmark and
operation, and feature-importance analysis showed the model relies most on
precisely the least-stable parameters. We further demonstrate that a zero-skill
model matches "99% accuracy," exposing accuracy as meaningless for this rare event.

These results establish a reproducible method for honestly evaluating operational
flare forecasts and caution that benchmark scores must not be read as real-world
performance.
