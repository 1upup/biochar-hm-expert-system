# Biochar-HM Expert System V3.9.6

A Streamlit research application for predicting metal immobilization in soils and exploring biochar design targets with empirical support assessment.

## Scope

The application includes seven interface sections: system introduction, experimental data management, automated modeling, single-sample prediction, inverse design, batch prediction, and prediction history. The AutoML candidates are XGBoost, Random Forest, LightGBM, and SVR. Additional analyses include SHAP/PDP, study-wise sensitivity, weighted differential evolution optimization, and empirical support calibration.

The original application source has been preserved byte-for-byte. Packaging does not change algorithms, page layouts, or colors.

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

Record the Python version in this README. Inspect the lock file for machine-specific paths before publishing it. Install the reviewed lock file for subsequent reproductions.

## Data and workflow

1. Open the experimental data management page (实验数据管理).
2. Upload a UTF-8/UTF-8-BOM CSV with the columns described in `data/README.md`.
3. Open the automated modeling page (自动化建模分析), choose and record the trial budget, and run modeling.
4. Run interpretation, calibration, prediction and inverse design sections as needed. Record settings for each paper figure and table.

**The built-in 600-record synthetic dataset is for functionality testing only. It is not the literature database and cannot reproduce manuscript findings.**

This package does not include the real manuscript dataset or paper results. Add the authorized dataset, or specify its exact location in the Supplementary Information or repository. Do not claim that this package reproduces numerical manuscript results until that workflow has been verified.

The application creates `biochar_history.db` locally. Existing personal history files and Python environment folders do not need to be published. Uploading this repository does not deploy a live website.

## Reproducibility

Record the dataset version and row order, Python and dependency versions, model choice, trial budgets, split protocol, analysis settings, scenario inputs, optimization bounds and objective weights. The source uses seed 42 in several procedures; a fixed seed alone does not guarantee identical results across dependency versions.

For study-wise analyses provide traceable Study_ID values; do not invent source identifiers. Study_ID and Source_Title are metadata, not predictors.

## Interpretation

Predictions and inverse-design outputs are research estimates, not validated remediation prescriptions. Empirical support scores describe support relative to the training data, not experiment success probabilities or proof of producibility.

## Validation status of this package

Source identity and Python syntax compilation checked. Full Streamlit execution, dependency installation and model-result reproduction have NOT been tested in this packaging task.

## License and citation

No license has been selected in this preparation package. The authors should agree on code reuse terms and add an appropriate LICENSE before describing the project as open source. Add the finalized article citation and a persistent code archive identifier when available.
