# Academic and Statistical Approaches to F1 Race Outcome Prediction

## Which statistical frameworks are used (Bayesian hierarchical, Plackett-Luce / rank-ordered logit, Bradley-Terry, Elo, survival)?

### Takeaway
The peer-reviewed literature is dominated by ranking models: rank-ordered logit / Plackett-Luce (often Bayesian and multilevel), paired-comparison (Bradley-Terry) models, and Elo-style online generalizations of Plackett-Luce. Most published ML (random forest, gradient boosting, SVM) work is in theses or preprints and is less rigorous. Survival models in the F1 literature model driver *careers*, not in-race DNFs.

### Cited Findings
- van Kesteren & Bergkamp (2023, JQAS 19(4):273-293): Bayesian multilevel rank-ordered logit fitted in Stan on 2014-2021 (160 races, 51 drivers, 19 constructors). Latent ability = long-run driver skill + seasonal driver form + long-run constructor advantage + seasonal constructor form, all zero-mean normal effects interpretable as log-odds of beating an average competitor (Elo-like) — [arXiv 2203.08489](https://ar5iv.labs.arxiv.org/abs/2203.08489); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10660124/)
- Henderson & Kirrane (2018, Bayesian Analysis 13(2):335-358): compare truncated and time-weighted Plackett-Luce models for *probabilistic forecasting* of F1 results, 2010-2013 seasons — [Project Euclid](https://projecteuclid.org/journals/bayesian-analysis/volume-13/issue-2/A-Comparison-of-Truncated-and-Time-Weighted-PlackettLuce-Models-for/10.1214/17-BA1048.full); [Newcastle eprints](https://eprints.ncl.ac.uk/231830)
- Powell (2023, JQAS, doi 10.1515/jqas-2023-0004): generalized Elo for multiplayer races, acting as an online estimator of Plackett-Luce parameters. Its key twist is treating a race as sequential *elimination* rounds (identifying the worst finisher first) rather than selecting the best ("endurance is better than speed"). Predictive power demonstrated on F1 data (numbers not retrieved) — [White Rose eprints](https://eprints.whiterose.ac.uk/200471)
- Bell, Smith, Sabel & Jones (2016, JQAS 12(2):99-112): cross-classified multilevel model (MCMC) of driver points, 1950-2014, with team, team-year and driver levels; effects allowed to vary by year and track type — [White Rose PDF](https://eprints.whiterose.ac.uk/id/eprint/96995/14/WRRO_96995.pdf); [RePEc](https://econpapers.repec.org/RePEc:bpj:jqsprt:v:12:y:2016:i:2:p:99-112:n:3)
- Anderson (2014, Applied Economics 46(15)): compares two paired-comparison (Bradley-Terry type) models and a rank-ordered logit on the 2012 F1 season; argues they beat points-based rankings — [RePEc](https://econpapers.repec.org/RePEc:taf:applec:v:46:y:2014:i:15:p:1778-1787)
- Dynamic Plackett-Luce working paper: driver and team ratings 1984-2019 with time-varying trajectories; finds team gaps widened over the last decade — [PDF](https://soichiroy.github.io/files/misc/dynamic_ranking_single.pdf)
- Hankin (2024, Journal of Statistical Software v109i08): generalized Plackett-Luce likelihoods (R package hyper2) with F1 among examples — [JSS](https://www.jstatsoft.org/article/download/v109i08/4581)
- Non-academic but influential: FiveThirtyEight F1 Elo rates teams and drivers separately and blends them for head-to-head odds — [FiveThirtyEight](https://fivethirtyeight.com/features/in-formula-one-does-the-driver-or-car-matter-more); Martin Ingram's blog: rank-ordered logit with a random-walk season prior and a separate Bernoulli DNF model — [blog](https://martiningram.github.io/f1-model)
- Survival analysis in F1 is about career length: Celik (2020, Social Science Quarterly) finds podium finishes lower the probability of exiting F1 and changing teams helps survival; Mourão (2018, Economic Analysis and Policy) uses Cox models on non-winning drivers — [RePEc Celik](https://econpapers.repec.org/RePEc:bla:socsci:v:101:y:2020:i:4:p:1271-1281); [RePEc Mourão](https://econpapers.repec.org/RePEc:eee:ecanpo:v:59:y:2018:i:c:p:54-68)
- ML-centric work is mostly theses or preprints: an ensemble of SVM + gradient boosting + random forest (JCSI, 2026) reporting that telemetry strongly affects quality; a Complutense thesis that combines ML with ranking models; a 2025 Preprints.org two-model system (not peer reviewed) — [JCSI](https://ph.pollub.pl/index.php/jcsi/article/download/8462/5371/41814); [UCM](https://docta.ucm.es/entities/publication/b048a690-9cab-47d0-aad6-c8f2764bdc94); [Preprints](https://www.preprints.org/manuscript/202504.1471)

### Inferences
- A Plackett-Luce / rank-ordered logit with driver and constructor strengths, plus grid position as a covariate, is the academically standard way to get joint, coherent podium probabilities. It is a natural upgrade from independent logistic regression.
- Powell's elimination-order Plackett-Luce suggests putting more weight on the *bottom* of the order. For podium prediction specifically, a top-truncated model (Henderson & Kirrane) may be more relevant.

### Gaps
- I could not retrieve the numeric results of Henderson & Kirrane or Powell (the full text was unavailable or the fetch failed).
- I found no peer-reviewed F1 paper that models in-race DNF hazard with survival methods. Ingram's blog (separate Bernoulli DNF) is the closest.

## How do they separate driver skill from car performance?

### Takeaway
They use additive driver and constructor (or team-year) effects in a hierarchical model, identified through teammate comparisons and driver moves between teams. The consistent finding is that the car explains most (about 86-88%) of the variance in results.

### Cited Findings
- van Kesteren & Bergkamp: the constructor explains about 88% of race-result variance (89% CI 0.775-0.945). Estimated variances: constructor advantage 2.65, constructor seasonal form 0.54, driver skill 0.29, driver form 0.12. Counterfactual: Hamilton in an Alfa Romeo has only a 0.36 probability of beating Räikkönen in a Mercedes — [arXiv](https://ar5iv.labs.arxiv.org/abs/2203.08489)
- Bell et al. (2016) found the comparable team share was about 86% for 1950-2014 — cited in [van Kesteren & Bergkamp](https://ar5iv.labs.arxiv.org/abs/2203.08489); original at [White Rose](https://eprints.whiterose.ac.uk/id/eprint/96995/14/WRRO_96995.pdf)
- FiveThirtyEight rates teams and drivers separately and blends them — [FiveThirtyEight](https://fivethirtyeight.com/features/in-formula-one-does-the-driver-or-car-matter-more)

### Inferences
- For 2022-2025 data, team/constructor form features should carry most of the signal. Driver effects are small and noisy. Teammate-relative features are the cleanest way to isolate driver skill, which matches the project plan.
- Season-level effects in these models are fitted on the full season, which leaks future races. Any use for forecasting needs filtered/online estimates (for example Elo or time-weighted Plackett-Luce).

### Gaps
- None of the retrieved papers report driver-vs-car shares specifically for the 2022+ ground-effect era.

## How do they model the race as a ranking (jointly) rather than as independent binaries?

### Takeaway
The models use Plackett-Luce / rank-ordered logit: the finishing order is generated by repeatedly choosing the next finisher with softmax over the remaining drivers' strengths. Truncated versions model only the top k, and time-weighted versions down-weight old races.

### Cited Findings
- The rank-ordered logit treats each race's full finishing order as the outcome. Strengths are log-odds of beating a competitor — [arXiv](https://ar5iv.labs.arxiv.org/abs/2203.08489)
- Henderson & Kirrane explicitly compare truncated (top-k only) and time-weighted Plackett-Luce variants for forecasting — [Project Euclid](https://projecteuclid.org/journals/bayesian-analysis/volume-13/issue-2/A-Comparison-of-Truncated-and-Time-Weighted-PlackettLuce-Models-for/10.1214/17-BA1048.full)
- Powell's elimination-type Plackett-Luce identifies the worst competitors first, updated online after each race — [White Rose](https://eprints.whiterose.ac.uk/200471)
- DNF handling: van Kesteren & Bergkamp dropped non-finishers (590 of 3,267 rows) and found results somewhat sensitive to this choice — [arXiv](https://ar5iv.labs.arxiv.org/abs/2203.08489). Ingram models DNFs as a separate Bernoulli process — [blog](https://martiningram.github.io/f1-model)

### Inferences
- Practical route for this project: fit a conditional logit (softmax over the 20 drivers per race) on linear features. P(podium) can then be obtained by Monte Carlo sampling of Plackett-Luce orders, or approximated by the softmax win probability. This guarantees that the probabilities sum sensibly within each race, which independent logistic regression does not.
- A two-part model, P(finish) × P(podium | finish), mirrors the DNF treatments found in the literature.

### Gaps
- No retrieved paper directly compared a joint ranking model against independent binary classifiers for podium prediction.

## What evaluation metrics and baselines (qualifying order, betting odds) are used, and how much do they beat them?

### Takeaway
Bayesian papers evaluate with LOO-CV / ELPD, posterior predictive checks and log scores, not hit rates against a grid baseline. I found no peer-reviewed comparison of F1 models against bookmaker odds. Grid-order baselines are rarely reported formally.

### Cited Findings
- van Kesteren & Bergkamp used LOO-CV (ELPD): basic -3993.36, circuit -3992.29, circuit+weather -3993.66, weather -3995.55, with differences within standard errors. They state the model is descriptive and not fit for forecasting because season effects are unknown in advance. Their 2021 simulation under-predicted points for top drivers — [arXiv](https://ar5iv.labs.arxiv.org/abs/2203.08489)
- Henderson & Kirrane frame their work as probabilistic forecasting (proper scoring) on 2010-2013 — [Project Euclid](https://projecteuclid.org/journals/bayesian-analysis/volume-13/issue-2/A-Comparison-of-Truncated-and-Time-Weighted-PlackettLuce-Models-for/10.1214/17-BA1048.full)
- A search for F1 betting market efficiency studies returned only horse racing, tennis and NHL favourite-longshot work, with no F1 study — e.g., [Southampton](https://eprints.soton.ac.uk/153443)
- A non-peer-reviewed portfolio project reported 63% winner precision for 2021 — via search summary of [felipesanze portfolio](https://felipesanze.github.io/portfolio/project-1)

### Inferences
- The project's "tie the grid top-3 baseline" result is consistent with the literature's message: grid position plus car strength captures most of the signal. Gains are likely to show up in Brier/log loss and calibration rather than in hit counts. Report the log score as well.
- Bookmaker odds (if obtainable pre-race) would be the strongest baseline. No academic benchmark exists to compare against.

### Gaps
- I found no quantified "beats qualifying order by X" figure in peer-reviewed work.
- No F1-specific betting-odds benchmark study was found.

## What is the predictive value of qualifying, practice pace, weather, track type and reliability?

### Takeaway
Grid position is the dominant single predictor: pole is worth about 2 finishing positions, or about +10 percentage points of win probability. Weather and track-type extensions did not significantly improve Bayesian fits. Reliability matters, but its treatment changes the results.

### Cited Findings
- Wesselbaum & Owen (2021, Australian Economic Review 54(1):164-173), 1950-2013: the pole sitter gains about two finishing positions, or about a 10 percentage point higher win probability, versus other grid slots — [vLex](https://au.vlex.com/vid/the-value-of-pole-862899004)
- Mühlbauer (2010) found starting grid position is the strongest determinant of outcomes. A 2025 Wharton Sports Analytics submission (Weissbock, not peer reviewed) argues that study used a truncated sample and may overstate grid relative to qualifying pace — summarized from [Weissbock WSA](https://wsb.wharton.upenn.edu/wp-content/uploads/2025/12/WEISSBOCK_WSA_F1-Submission.pdf)
- McCarthy & Rotthoff (2013): in 2012, mid-grid (P10) and last positions had the highest collision probability — as cited in [Weissbock WSA](https://wsb.wharton.upenn.edu/wp-content/uploads/2025/12/WEISSBOCK_WSA_F1-Submission.pdf)
- Wet (17 of 160 races) and street-circuit extensions did not clearly improve LOO fit in van Kesteren & Bergkamp — [arXiv](https://ar5iv.labs.arxiv.org/abs/2203.08489)
- Bell et al. let team and driver effects vary by track type — [White Rose](https://eprints.whiterose.ac.uk/id/eprint/96995/14/WRRO_96995.pdf)
- Practice-to-qualifying prediction appears in an AWS ML blog and an Erdos Institute 2024 project (not peer reviewed) — [AWS](https://aws.amazon.com/blogs/machine-learning/predicting-qualification-ranking-based-on-practice-session-performance-for-formula-1-grand-prix/); [Erdos](https://www.erdosinstitute.org/project-database/may-summer-2024/forecasting-outcomes-in-formula-1-racing)
- The JCSI ensemble paper reports that telemetry data substantially affects prediction quality — [JCSI](https://ph.pollub.pl/index.php/jcsi/article/download/8462/5371/41814)

### Inferences
- Since grid already encodes car and driver pace, the improvements most likely to beat grid top-3 are: (1) quali gap-to-pole in seconds or percent (a continuous pace signal beyond rank), (2) team race-pace form relative to quali form (some cars race better than they qualify), (3) a reliability/DNF prior per team, and (4) joint softmax modelling. Weather is unlikely to help much given how few wet races there are.

### Gaps
- I could not extract Weissbock's quantitative grid-vs-pace results (the PDF was not parsed).
- I found no rigorous study quantifying the added value of practice long-run pace for race prediction.
