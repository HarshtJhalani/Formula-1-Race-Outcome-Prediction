# Open-source / hobbyist F1 race outcome prediction projects

## What do notable projects use (data, timing, features, model, split, accuracy, baseline, leakage)?

### Takeaway
Most hobbyist projects are FastF1/Ergast/OpenF1 + Random Forest/XGBoost with qualifying position plus rolling form. Few report a grid baseline, and several use random splits or race-day info. The best-documented one (f1-prediction-platform) uses walk-forward backtesting and only modestly beats a grid-history baseline.

### Cited Findings
- **f1-prediction-platform (weighted-v4):** 7 features fitted on 2010–2026 data (weights per SD): grid (−log slot) +0.64, grid × (1 − circuit overtake rate) +0.52, team race pace +0.57, driver points +0.56, team quali pace +0.40, quali pace +0.36, teammate edge +0.14. Walk-forward, leakage-free backtest over 190 races (2018–2026), refit before each race. v4: log loss 1.208, Brier 0.566 (multi-class), top-1 60.0%, winner-in-top-3 90.0%. Grid-history baseline (P(win | grid slot) from past races, so the pole-sitter is always the pick): 1.457 log loss, top-1 55.3%, top-3 87.9%. v4's log-loss gain is 0.249 ± 0.099 (2 SE). An earlier v3 lost to "pole-sitter wins." — [f1-prediction-platform docs](https://f1-prediction-platform-web.pages.dev/docs/prediction-model)
- **Rhianne G race predictor:** OpenF1 data 2023–2025 (1,139 driver-race rows). Features: qualifying position (~22% importance), last-5-race rolling position, team season-average position and win/podium rates, driver circuit history, career stats computed from earlier races only. Compared LR, RF and GB, picked RF. Time split: train 2023–24, test 2025. Reports 91.0% podium accuracy and AUC 0.967 on 2025, but no numeric grid baseline. Inconsistent test size (200 vs 220). A "season-average team position" feature may include future races (leakage risk). — [Rhianne's Portfolio](https://rhianneg.github.io/projects/project1_Race-Predictor/race-predictor.html)
- **Frank Ndungu blog:** 2014–2024 data from unnamed APIs. Features: quali position, driver last-5 form, team last-5 points, circuit score, weather. RandomForestClassifier (100 trees, depth 10) beat GB, NN and LR. **Random 80/20 `train_test_split`** (leakage). Reports podium 65%, top-10 75%, winner 45%, with metrics undefined and no baseline. — [frank-ndungu.vercel.app](https://frank-ndungu.vercel.app/blog/f1-prediction-models)
- **mar-antaya/2025_f1_predictions (~560 stars):** a gradient boosting regressor on lap times, using 2024 race results, 2025 qualifying, weather forecasts and FastF1. It predicts race pace/lap time, then ranks drivers; a mirror reports 3.22 s MAE. It was widely covered in the press, but has no formal backtest or baseline. — [Techstrong](https://techstrong.ai/ai-at-the-edge/data-scientist-uses-ai-to-predict-f1-winners); [Motorsport.com](https://www.motorsport.com/f1/news/fans-are-using-ai-to-predict-f1-race-results-and-the-software-is-only-getting-smarter/10717037/); [mirror README](https://github-users.yhajoa.workers.dev/melisasvr/2025_f1_predictions); [relatedrepos](https://relatedrepos.com/gh/mar-antaya/2025_f1_predictions)
- **Other FastF1 podium repos** (README-level only, unverified): sarthakasap11/f1podiumpredictor (two RFs, one for podium and one for winner, 3+ seasons, Streamlit); Ravencrest-01/F1-Model (Ergast 2000–2024 + FastF1 2023–25); BaghettiGH/F1-Podium-Prediction (logistic regression); TomandJerry0811/Formula-1-prediction (RF + XGBoost, FastF1 2018–2024, self-reported "85%+" top-3 accuracy). — [search summary sources: ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fbaghettigh%2Ff1-podium-prediction); [Ravencrest mirror](https://github.ddaodan.workers.dev/Ravencrest-01/F1-Model); [HF](https://huggingface-proxy.pages.dev/sarthakasap11/f1podiumpredictor); [Mintlify TomandJerry](https://www.mintlify.com/TomandJerry0811/Formula-1-prediction)
- **jaeyow/f1-predictor:** Ergast-based, with weather from Wikipedia/World Weather Online, plus home circuit, age/experience and qualifying. Tries multiple classifiers and regressors; the README gives no split or baseline. — [GitHub](https://www.github.com/jaeyow/f1-predictor)
- **Leakage handling seen in projects:** one replaced post-race points with cumulative points before the race; a GWU presentation excluded lap times/laps completed as leakage, and found **the grid P1–3 baseline outperformed their model**. — [heyisula/f1 mirror](https://proxy-ga.blitzz.co/proxy/123456/github.com/heyisula/f1); [GWU presentation](https://datasci.columbian.gwu.edu/sites/g/files/zaxdzs4746/files/2023-02/f1-final-presentation.pdf)
- **Data source note:** the Ergast API sunset after 2024 (Jolpica is the successor); legacy data is on Kaggle. — [info-523 project](https://info-523-f25.github.io/final-project-team-1/)

### Inferences
- High "accuracy" claims (85–91%) are mostly per-driver binary accuracy, where ~85% is achievable by predicting "no podium" for everyone (17/20 drivers). They are not comparable to hits-per-race.
- Only the walk-forward platform reports an honest baseline, and its edge over grid is real but modest. That is consistent with our model only tying grid top-3.

### Gaps
- No public project found that reports podium hits per race against a grid-top-3 baseline in the 2022+ era.
- Kaggle notebooks and Reddit threads were not reviewed in depth (tool budget).

## Features beyond grid; FP2 long-run pace

### Takeaway
The most useful extra features in documented work are circuit overtaking difficulty interacted with grid, team race pace, and teammate edge. FP2 long runs were tried by at least one rigorous project and dropped as noise.

### Cited Findings
- The grid × (1 − overtake rate) "processional" term is the second-largest weight in v4. FP2 stint medians per compound are ingested but "practice long runs were tried for v4 and left out" (no gain). — [f1-prediction-platform](https://f1-prediction-platform-web.pages.dev/docs/prediction-model)
- A betting-analytics method for FP2 long runs: stints of at least 5 laps, mean lap adjusted for compound and linear degradation, with fuel estimated from stint length. — [pillarlab](https://pillarlab-web.pages.dev/pillars/practice-pace-vs-market-line)
- Academic (2006–2023, 7,800 entries): qualifying ordinal-regression coefficient 1.558, about 4x any practice session (FP3 0.379, FP2 0.368, FP1 0.291). Spearman of start grid vs finish was 0.748 for 2010–23. — [arXiv 2507.10966](https://arxiv.org/html/2507.10966v1)
- Weather, home circuit and age/experience are common candidates; actual race-day weather is leakage unless it comes from a forecast. — [jaeyow](https://www.github.com/jaeyow/f1-predictor); [Ndungu](https://frank-ndungu.vercel.app/blog/f1-prediction-models)

### Inferences
- Practice pace is largely already captured by qualifying in post-quali prediction; track overtaking difficulty and a DNF/reliability model are the more promising additions.

### Gaps
- No public backtest isolating FP2 long-run value in the 2022–2025 era.

## Model types and ranking vs binary classification

### Takeaway
Hobbyists overwhelmingly use independent per-driver classifiers (RF/XGBoost/LR) or lap-time regressors. Ranking approaches (rank-ordered logit/Plackett-Luce, Elo) appear in stats blogs and academic work, usually without predictive backtests.

### Cited Findings
- Martin Ingram: rank-ordered logit (BTL), skill = driver + car with a random walk across seasons, DNFs modelled separately as Bernoulli, in Stan, on Ergast 2014+. No hold-out evaluation. — [blog](https://martiningram.github.io/f1-model)
- Powell (2023) generalises Elo to Plackett-Luce for races with online probabilistic forecasts; Bayesian work found reverse-PL fits F1 orders better. — [York preprint](https://www-users.york.ac.uk/~bp787/Generalizing_Elo_arxiv.pdf); [arXiv 2002.05953](https://arxiv.org/pdf/2002.05953)
- Dynamic Plackett-Luce for F1 drivers/teams 1984–2019. — [Yamauchi](https://soichiroy.github.io/files/misc/dynamic_ranking_single.pdf)
- eloF1: a simple pseudo-Elo (base 1000, adjusted vs average finish). — [GitHub](https://github.com/Shuvam586/eloF1)
- The platform's v4 normalises scores into a per-race softmax-style win distribution (multi-class log loss). — [f1-prediction-platform](https://f1-prediction-platform-web.pages.dev/docs/prediction-model)

### Inferences
- Per-race normalisation (probabilities summing to 3 for podium, or a PL model) is the main methodological difference from our independent logistic regression, and should improve Brier/calibration.

### Gaps
- No found hobbyist LambdaMART podium model with a reported backtest.
