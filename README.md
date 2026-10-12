# Formula 1 Race Podium Prediction

## Overview

This project aims to develop a machine learning pipeline to predict Formula 1 race podium finishers using historical race results, qualifying performance, and driver-level features.

The prediction will be made after qualifying and before the race begins, using only information available at that point.

The project currently focuses on historical data extraction, preprocessing, and initial feature engineering.

## Objectives

* Extract historical Formula 1 qualifying and race results.
* Build a structured dataset containing driver-level information.
* Engineer features using qualifying performance and historical driver results.
* Develop a machine learning model to predict podium finishers.
* Evaluate predictions using historical races that were not used for training.

## Technology Stack

* Python
* FastF1
* Pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* pytest

## Project Structure

```text
formula-1-race-podium-prediction/
├── data/
│   ├── cache/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── qualifying.py
│   ├── race_results.py
│   └── features.py
├── tests/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Implemented Components

### 1. Qualifying Data Extraction

Implemented a reusable extraction function using FastF1 to retrieve qualifying session results for a specified season and Grand Prix.

The extracted data includes available driver identifiers, driver names, constructor information, qualifying positions, and qualifying session times.

Results are exported to CSV files in `data/raw/`.

### 2. Race Results Extraction

Implemented a reusable extraction function to retrieve historical race results.

The extracted data includes driver information, constructor details, and actual finishing positions.

A binary podium target is generated:

* `1`: Finished in positions 1–3.
* `0`: Finished outside the top three.
* Missing: Finishing position is unavailable.

Missing finishing positions are preserved rather than automatically classified as non-podium results.

### 3. Initial Feature Engineering

Implemented a feature engineering module that merges qualifying and race datasets using season, Grand Prix, and driver identifiers.

The processed dataset contains driver information, qualifying position, finishing position, and podium targets.

The pipeline exports the resulting dataset to `data/processed/`.

### 4. Dataset Inspection

Added output summaries to inspect the processed dataset, its columns, and the distribution of podium outcomes.

## Current Dataset

The initial pipeline has been tested using the 2023 Singapore Grand Prix.

The extracted dataset contains 20 driver records, including three podium finishers and one driver with a missing finishing position.

This dataset validates the extraction and merging pipeline. It is not sufficient on its own to train or evaluate a reliable predictive model.

## Planned Improvements

* Extract data across multiple historical seasons and Grand Prix events.
* Investigate and include qualifying lap-time features.
* Engineer historical driver-performance features using only information available before each race.
* Develop a machine learning pipeline for podium prediction.
* Evaluate predictions using chronological train-test splits.
* Prevent data leakage during feature engineering and model evaluation.
* Generate ranked predictions for first, second, and third place.
* Add automated tests and improve pipeline reliability.

## Data Leakage Prevention

The project is designed to predict race outcomes using information available after qualifying and before the race begins.

Actual finishing positions are used to construct prediction targets and evaluate predictions, not as input features for the same race.

Historical performance features will be calculated using only races completed before the event being predicted.

## Setup

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the qualifying extraction module:

```powershell
python -m src.qualifying
```

Run the race results extraction module:

```powershell
python -m src.race_results
```

Run the feature engineering module:

```powershell
python -m src.features
```

## Project Status

In progress.

Historical qualifying and race results extraction, initial podium target generation, and dataset merging have been implemented and tested on one Grand Prix.

Historical feature engineering, model development, and predictive evaluation remain to be completed.
