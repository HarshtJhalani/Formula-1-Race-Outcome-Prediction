# Industry and Market Approaches to F1 Race Outcome Prediction

Scope note: Limited search budget (~8 searches/fetches). Many claims come from search-result summaries rather than full-text reads; these are flagged. No peer-reviewed F1-specific betting-efficiency study was found.

## How does AWS/F1 Insights build its race predictions and what inputs does it use?

### Takeaway
F1 Insights is mostly *in-race, live* prediction (gap/overtake probability, undercut threat, pit windows), built on XGBoost models trained in SageMaker on historical timing, tyre and weather data and served in Lambda on live timing. AWS publishes no accuracy figures, and no public source details the features of a pre-race "win probability" model.

### Cited Findings
- AWS ML blog: Pit Strategy Battle graphic predicts the time gap between two cars and overtake probability; also flags undercut/overcut as they unfold. — [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/accelerating-innovation-how-serverless-machine-learning-on-aws-powers-f1-insights)
- Inputs: live timing/position via API; historical timing, tyre and weather data for training and feature extraction; race state (driver positions, features) stored in DynamoDB. — [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/accelerating-innovation-how-serverless-machine-learning-on-aws-powers-f1-insights)
- Model: XGBoost (gradient boosted trees), trained in SageMaker, converted to a smaller format to run inside Lambda; target latency under 500 ms. KPIs named (share of predictions within a boundary, RMSE) but no values given. No Monte Carlo mentioned. — [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/accelerating-innovation-how-serverless-machine-learning-on-aws-powers-f1-insights)
- Insights list includes exit speed, predicted pit stop strategy, pit window, battle forecast, pit strategy battle, tyre performance; F1 cites 300+ sensors per car and ~1.1M data points/sec. — [Help Net Security (2021)](https://www.helpnetsecurity.com/2021/04/19/aws-formula-1/); [AWS F1 page](https://aws.amazon.com/sports/f1/)

### Inferences
- F1 Insights is a live, per-battle product, not a pre-race podium forecaster, so it is not a direct benchmark for a post-qualifying model. The useful lesson is the method: gradient boosting on historical timing, tyre and weather data.

### Gaps
- No public feature list, training labels or calibration for any AWS "win probability" graphic.
- I found no published accuracy numbers for any F1 Insights graphic.

## How do teams and analysts estimate race pace from FP2 long runs? Common FastF1 methods?

### Takeaway
Teams and analysts model lap time as base pace + fuel-mass effect + tyre degradation (often linear per stint), with laps filtered for traffic and outliers. Fuel loads in practice are unknown, which makes FP2 long-run comparisons noisy. Teams rely on Monte Carlo race simulators with probabilistic events.

### Cited Findings
- FastF1 docs' data analysis guide shows per-stint degradation as a linear fit of lap time vs lap number (slope = s/lap). It applies no fuel correction. — [FastF1 guide (mintlify mirror)](https://mintlify.com/theOehrly/Fast-F1/guides/data-analysis)
- Common fuel correction: subtract a fixed per-lap fuel penalty scaled by laps remaining (fuel-corrected lap time). — [R-bloggers, Visualising F1 Stint Strategies](https://www.r-bloggers.com/2016/04/visualising-f1-stint-strategies/)
- 2025 arXiv paper using FastF1 models lap time as a function of fuel mass plus a latent tyre-pace state, with pit stops as state resets (case study: Hamilton, 2025 Austrian GP). — [arXiv 2512.00640](https://arxiv.org/html/2512.00640v1) (summary only, not full read)
- Forum discussion: long runs reveal true pace better than single laps, but long runs on different fuel loads can differ by up to ~3 s, so runs need anchoring to a known fuel load. — [F1Technical forum](https://f1technical.net/forum/viewtopic.php?p=680505) (forum opinion)
- Mainstream long-run analyses (e.g., The Race / Mark Hughes) compare practice long-run averages between teams. — [The Race, Barcelona long-run analysis](https://www.the-race.com/formula-1/is-mclarens-return-to-form-real-barcelona-f1-long-run-analysis.md); [The Race, Mark Hughes Bahrain 2024](https://www.the-race.com/formula-1/mark-hughes-f1-2024-bahrain-gp-mercedes-red-bull.md) (not fetched)
- TracingInsights publishes FastF1-based analyses. — [TracingInsights Substack](https://tracinginsights.substack.com) (not fetched)
- Team practice: Monte Carlo simulation is described by a team strategist as the starting point for strategy; rain raises accident/SC likelihood; teams study historical SC timing by circuit. — [F1Technical news, How do teams define race strategies?](https://www.f1technical.net/news/25828) (summary only)
- Academic: Heilmeier et al. (TUM, 2020, Applied Sciences) add Monte Carlo probabilistic events (accidents, failures, FCY/SC, driver behaviour) to a lap-time race simulation to test strategy robustness. A follow-up "Virtual Strategy Engineer" uses neural networks with MC runs to output finishing-position distributions. — [MDPI Appl. Sci. 10(12):4229](https://www.mdpi.com/2076-3417/10/12/4229/html); [MDPI 10(21):7805](https://mdpi.com/2076-3417/10/21/7805)
- Monte Carlo strategy software reportedly became central in F1 around 1998. — [Autosport](https://www.autosport.com/f1/news/why-f1-strategic-thinking-must-encompass-more-than-pitstop-timings/10628623/) (summary only)

### Inferences
- A practical FastF1 FP2 feature would be: take each driver's longest FP2 stint, drop in/out laps and outliers (for example laps more than 107% of the stint median, or flagged with `IsAccurate`/yellow flags), apply a fuel correction (commonly ~0.03 s/lap per kg × ~1.5–1.8 kg/lap; this is a community rule of thumb with no source found here, so it needs calibrating), then compute median corrected pace and a degradation slope per compound, normalised to the session-best team.
- Caveats for the project: sprint weekends have only one practice session (no representative FP2), FP2 is sometimes wet or red-flagged, and unknown fuel or engine modes add noise. Expect lots of missing values. This should still be leakage-safe, since FP2 happens before the race.

### Gaps
- No published, validated fuel-correction constant for FastF1 data was found.
- Teams' internal simulator accuracy is not public.
- Heilmeier paper's SC probability numbers by circuit were not verified (full text not read).

## How accurate are betting markets at predicting F1 podiums vs the starting grid?

### Takeaway
I found no peer-reviewed study of F1-specific betting-market efficiency or calibration for podium markets. General sports-betting literature shows markets are broadly efficient but show a favourite-longshot bias. Applying that to F1 is an assumption.

### Cited Findings
- Horse racing (Snowberg & Wolfers, NBER w15923, ~5.6M starts): betting 100/1+ longshots loses ~61%, random betting ~23%, betting favourites ~5.5%. They attribute the bias to misperceived probabilities. — [NBER w15923](https://www.nber.org/system/files/working_papers/w15923/w15923.pdf)
- Bias is reported as more severe in bookmaker markets than pari-mutuel markets. — [J. Prediction Markets 2(2)](https://0-doi-org.brum.beds.ac.uk/10.5750/jpm.v2i2.440) (summary only)
- Methods such as Shin, power and goto-conversion de-vig odds into probabilities. — [Wikipedia: Favourite-longshot bias](https://wikipedia.com/wiki/Favourite-longshot_bias)
- Small samples make it hard to tell genuinely efficient markets from ones that only appear efficient (football, simulation study). This is relevant given F1's ~22–24 races per season. — [Uni Bielefeld, Are Betting Markets Inefficient?](https://pub.uni-bielefeld.de/record/2983980)
- F1.com publishes pre-race odds from F1's official betting data supplier (for example, 2024 Hungarian GP team win odds of McLaren 1.495, Red Bull 8.66). — [Formula1.com betting guide](https://www.formula1.com/en/latest/article/betting-guide-who-are-the-favourites-as-f1-moves-on-to-hungary.MDxTg5ThdAlyver4kIe21)

### Inferences
- Post-qualifying de-vigged podium odds would make the strongest practical benchmark (they price in grid, pace, reliability and weather). Historical F1 odds archives (Betfair, Oddschecker) would have to be sourced manually.

### Gaps
- No F1 podium-odds calibration or efficiency study was found.
- No dataset comparing odds against the grid-top-3 baseline was found.

## What is the realistic ceiling for pre-race podium prediction given DNFs, safety cars, weather?

### Takeaway
The grid-top-3 baseline is hard to beat. Hobby and student projects report that grid position dominates other features and that adding pit or circuit features barely helps. No authoritative ceiling exists. Pole-to-win conversion is ~40–45% historically, which shows how much randomness remains.

### Cited Findings
- Student project: predicting grid P1–P3 as the podium was exactly right in 28.6% of races, and at least one grid top-3 driver podiumed in 100% of races in their sample. Their Elastic Net model did not beat this, and pit/circuit features had near-zero effect once grid was included. — [GWU F1 final presentation](https://datasci.columbian.gwu.edu/sites/g/files/zaxdzs4746/files/2023-02/f1-final-presentation.pdf) (summary only; sample and era unverified)
- LightGBM podium classifier, TimeSeriesSplit CV, ~1,500 driver-race rows: ~56.7% CV podium accuracy, with a note on wide confidence intervals. — [Box Box project](https://moukthika-gunapaneedu.github.io/box-box/model/) (hobby; metric definition differs)
- Other projects claim 76–89.5% "accuracy", but with undefined or per-driver metrics that are not comparable. — [TomandJerry0811 model performance](https://mintlify.com/TomandJerry0811/Formula-1-prediction/resources/model-performance)
- Pole has won ~43.2% of all championship GPs historically. — [Formula Report Substack](https://f1chronicle.substack.com/p/how-often-does-the-pole-sitter-win)

### Inferences
- With an "out of 3" hit metric, a grid baseline probably scores ~2/3 per race on average in 2022–2025. That figure is not sourced and should be computed from the project's own data. Gains above it will likely be small, around 0.1–0.2 hits per race, with Brier/log-loss improvements easier to show than hit-count gains.
- DNFs, first-lap incidents, SC timing and rain are largely unpredictable from pre-race data, so a model can at best spread probability, not reach certainty.

### Gaps
- No peer-reviewed estimate of the irreducible error for F1 podium prediction.
- No sourced 2022–2025 DNF rate or SC-per-race rate by circuit was found in this pass.

## What factors beyond grid position do practitioners say matter most?

### Takeaway
Practitioners point to circuit overtaking difficulty (seen in pole conversion), race-pace vs qualifying-pace gaps (long runs and degradation), strategy and undercut power, and safety-car/weather probability. Circuit pole-conversion rates vary widely, which supports a track-level "grid importance" feature.

### Cited Findings
- Monaco: pole won 15 of 21 since 2004 (71.4%). — [GPFans](https://www.gpfans.com/en/f1-news/1085387/f1-monaco-grand-prix-wins-from-pole-position-alltime-stats/)
- Mexico ~45% pole-to-win; Interlagos 17/40 races won from pole. — [GPblog Mexico](https://www.gpblog.com/en/news/98048/stats-how-valuable-is-pole-position-for-the-mexican-grand-prix.html); [Lights Out blog](https://lightsoutblog.com/f1-grid-slot-stats-sao-paulo-grand-prix-interlagos)
- Low-confidence source lists Monza 33.3%, Silverstone 38.3%, Spa 39.7%, Hungaroring 37.5% pole conversion, with no methodology given. — [rpx-ee2 insights](https://rpx-ee2.pages.dev/insights/pole-position.html)
- Driver-level pole conversion differs widely (Leclerc ~18.5%; Verstappen 77–84% depending on the source). — [GPblog](https://www.gpblog.com/en/news/245886/verstappen-statistics-f1-pole-positions-charles-leclerc.html); [Sportskeeda](https://www.sportskeeda.com/f1/top-5-f1-drivers-best-pole-position-win-ratio); [F1oversteer](https://www.f1oversteer.com/features/the-best-pole-to-win-ratios-in-formula-1-history-as-five-drivers-achieve-100-records/)
- Team strategists: rain raises accident and SC probability; historical SC timing is studied per circuit. — [F1Technical](https://www.f1technical.net/news/25828)
- AWS F1 Insights treats tyre performance, pit windows and undercut threat as key race determinants. — [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/accelerating-innovation-how-serverless-machine-learning-on-aws-powers-f1-insights)

### Inferences
- Leakage-safe candidate features: (1) a circuit "grid-stickiness" index computed only from *prior* seasons' grid-to-finish correlation at that track; (2) FP2 fuel-corrected long-run pace rank vs qualifying rank (the race-pace minus quali-pace gap); (3) a rolling team/driver DNF rate with shift(1); (4) a rain forecast or wet-quali flag. Interactions such as grid × track stickiness may help logistic regression.
- 2026 regulations (new power units with ~50/50 ICE/electric split, active aero, and the Overtake mode replacing DRS) change overtaking and degradation dynamics. Track-level indices learned on 2022–2025 data probably won't transfer, which is consistent with the project excluding 2026. The regulation details come from general knowledge and were not sourced in this pass.

### Gaps
- No standardized, published overtaking-difficulty index per circuit was found.
- No sourced per-circuit SC probability table for 2022–2025.
