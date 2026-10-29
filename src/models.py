"""Train logistic regression, tune C on 2024, compare to the grid baseline."""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import baseline_prob, evaluate, fit_grid_baseline, load_features, time_split
from src.features import MODEL_FEATURES

C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]


def make_logreg(C):
    # Imputer: form features are NaN in a driver's/team's first race
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000),
    )


def tune_logreg(train, valid):
    """Pick C by Brier on validation (smoother than hits)."""
    scores = {}
    for C in C_GRID:
        model = make_logreg(C).fit(train[MODEL_FEATURES], train["Podium"])
        scored = valid.assign(Prob=model.predict_proba(valid[MODEL_FEATURES])[:, 1])
        scores[C] = evaluate(scored, "Prob")
    best_C = min(scores, key=lambda c: scores[c]["brier"])
    return best_C, scores


def main():
    train, valid = time_split(load_features())

    valid["BaselineProb"] = baseline_prob(valid, fit_grid_baseline(train))
    best_C, scores = tune_logreg(train, valid)
    print(pd.DataFrame(scores).T)

    model = make_logreg(best_C).fit(train[MODEL_FEATURES], train["Podium"])
    valid["LogRegProb"] = model.predict_proba(valid[MODEL_FEATURES])[:, 1]
    print(pd.DataFrame({
        "baseline (grid top 3)": evaluate(valid, "BaselineProb"),
        f"logreg (C={best_C})": evaluate(valid, "LogRegProb"),
    }).T)


if __name__ == "__main__":
    main()
