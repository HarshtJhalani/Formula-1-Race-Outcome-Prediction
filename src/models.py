"""Train logistic regression, tune C on 2024, compare to the grid baseline."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import baseline_prob, evaluate, fit_grid_baseline, load_features, time_split
from src.features import MODEL_FEATURES

C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "podium_logreg.joblib"


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


def save_final_model(C=0.3):
    """Train on all 2022-2025 data and save the model with its metadata."""
    train, valid, test = time_split(load_features(), include_test=True)
    fit_set = pd.concat([train, valid, test], ignore_index=True)
    model = make_logreg(C).fit(fit_set[MODEL_FEATURES], fit_set["Podium"])

    # Scores from the honest evaluation (fit on 2022-24, scored on 2025 once)
    eval_model = make_logreg(C).fit(pd.concat([train, valid])[MODEL_FEATURES], pd.concat([train, valid])["Podium"])
    test["Prob"] = eval_model.predict_proba(test[MODEL_FEATURES])[:, 1]

    bundle = {
        "model": model,
        "features": MODEL_FEATURES,
        "C": C,
        "trained_on": sorted(fit_set["Year"].unique().tolist()),
        "n_rows": len(fit_set),
        "test_2025": {k: float(v) for k, v in evaluate(test, "Prob").items()},
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved {MODEL_PATH} (trained on {bundle['trained_on']}, 2025 test {bundle['test_2025']})")
    return bundle


def load_final_model():
    return joblib.load(MODEL_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="train on 2022-2025 and save the final model")
    if parser.parse_args().save:
        save_final_model()
        return

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
