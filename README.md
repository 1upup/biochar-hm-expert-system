# Biochar-HM Expert System V3.9.6

A Streamlit research application for predicting metal immobilization in soils and exploring biochar design targets with empirical support assessment.

## Scope

The application includes seven interface sections: system introduction, experimental data management, automated modeling, single-sample prediction, inverse design, batch prediction, and prediction history. The AutoML candidates are XGBoost, Random Forest, LightGBM, and SVR. Additional analyses include SHAP/PDP, study-wise sensitivity, weighted differential evolution optimization, and empirical support calibration.


## Setup and launch (Windows Command Prompt)

Use the Python environment in which the original application works, where possible. For a new isolated environment:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m streamlit run biochar_hm_v3_9_6.py
```

If `python` is unavailable but the Windows launcher is installed, use `py -m venv .venv` for the first command. After activation, use `python`.

`requirements.txt` lists direct dependencies identified from imports; it is not a validated version lock. Verify compatibility locally. After confirming the workflow in a dedicated environment, record:

```bat
python --version
python -m pip freeze > requirements-lock.txt
```

Python version:3.7.3

## Data and workflow

1. Open the experimental data management page (实验数据管理).
2. Upload a UTF-8/UTF-8-BOM CSV with the columns described in `data/README.md`.
3. Open the automated modeling page (自动化建模分析), choose and record the trial budget, and run modeling.
4. Run interpretation, calibration, prediction and inverse design sections as needed. Record settings for each paper figure and table.

**The built-in 600-record synthetic dataset is for functionality testing only. It is not the literature database and cannot reproduce manuscript findings.**

This package does not include the real manuscript dataset or paper results. Add the authorized dataset, or specify its exact location in the Supplementary Information or repository.

The application creates `biochar_history.db` locally. Existing personal history files and Python environment folders do not need to be published. Uploading this repository does not deploy a live website.

## Reproducibility

Record the dataset version and row order, Python and dependency versions, model choice, trial budgets, split protocol, analysis settings, scenario inputs, optimization bounds and objective weights. The source uses seed 42 in several procedures; a fixed seed alone does not guarantee identical results across dependency versions.

For study-wise analyses provide traceable Study_ID values; do not invent source identifiers. Study_ID and Source_Title are metadata, not predictors.

## Interpretation

Predictions and inverse-design outputs are research estimates, not validated remediation prescriptions. Empirical support scores describe support relative to the training data, not experiment success probabilities or proof of producibility.

## Validation status

The application has been successfully run in the authors' local environment, and its main workflows have been checked. The tested Python and dependency versions are documented in this repository. Reproduction of the numerical results reported in the manuscript has not yet been verified.

## License and citation

