# Biochar-HM Expert System V3.9.6
# Run with: streamlit run biochar_hm_v3_9_6.py
# Dependencies: streamlit pandas numpy optuna shap seaborn matplotlib scipy
#               scikit-learn>=1.2 xgboost lightgbm

import streamlit as st
import pandas as pd
import numpy as np
import optuna
import shap
import seaborn as sns
import io
import time
import datetime
import sqlite3
import hashlib
import re
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from scipy.stats import spearmanr, kruskal

# Journal-inspired figure theme.
# Categorical hues draw on the ggsci NPG/nrc palette, inspired by
# Nature Reviews Cancer: https://nanx.me/ggsci/reference/pal_npg.html
# Figure styling follows the Nature research figure guide:
# https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/
# Edit these constants to change all figures consistently.
PAPER_PALETTE = [
    "#3C5488",  # deep blue
    "#E64B35",  # vermilion
    "#00A087",  # teal
    "#8491B4",  # slate violet
    "#4DBBD5",  # cyan
    "#7E6148",  # brown
    "#F39B7F",  # salmon
    "#91D1C2",  # pale teal
    "#B09C85",  # taupe
    "#DC0000",  # red
]

COLOR_BLUE = "#3C5488"
COLOR_ORANGE = "#D99023"
COLOR_PINK = "#E64B35"
COLOR_PURPLE = "#8491B4"
COLOR_LIGHT_BLUE = "#4DBBD5"
COLOR_OLIVE = "#00A087"
COLOR_DARK = "#262B33"
COLOR_GRID = "#DCE2E8"
COLOR_LIGHT = "#F5F7FA"
COLOR_REFERENCE = "#727B86"

# Continuous data use dedicated maps, rather than a categorical color sequence.
SHAP_CMAP = plt.get_cmap("coolwarm")
HEATMAP_CMAP = plt.get_cmap("Blues")
DIVERGING_CMAP = plt.get_cmap("RdBu_r")

# Include display aliases so baseline and tuned-model figures share model colors.
MODEL_COLOR_MAP = {
    "XGBoost": COLOR_BLUE,
    "RandomForest": COLOR_OLIVE,
    "Random Forest": COLOR_OLIVE,
    "LightGBM": COLOR_PINK,
    "SVR": COLOR_PURPLE,
    "Linear Regression": COLOR_REFERENCE,
}
METAL_COLOR_MAP = {
    "Cd": COLOR_BLUE,
    "Pb": COLOR_PINK,
    "Cu": COLOR_OLIVE,
    "Zn": COLOR_PURPLE,
    "As": COLOR_ORANGE,
    "Cr": COLOR_LIGHT_BLUE,
    "Mn": "#7E6148",
    "Ni": "#91D1C2",
    "Hg": "#B09C85",
}
PH_COLOR_MAP = {
    "Acidic (<6.5)": COLOR_PINK,
    "Neutral (6.5-7.5)": COLOR_OLIVE,
    "Alkaline (>=7.5)": COLOR_BLUE,
}
RBIO_COLOR_MAP = {
    "Low": "#C6DBEF",
    "Medium": "#6BAED6",
    "High": "#2171B5",
}
FEATURE_FAMILY_COLORS = {
    "Soil": COLOR_BLUE,
    "BC": COLOR_OLIVE,
    "Metal": COLOR_PINK,
}


def make_palette_dict(order):
    """Keep each known metal's color stable when a dataset omits other metals."""
    colors = {}
    for label in order:
        if label in METAL_COLOR_MAP:
            colors[label] = METAL_COLOR_MAP[label]
        else:
            # Deterministic fallback also stays stable across data subsets.
            index = int(hashlib.sha256(str(label).encode("utf-8")).hexdigest()[:8], 16)
            colors[label] = PAPER_PALETTE[index % len(PAPER_PALETTE)]
    return colors


def feature_color(feature):
    """Use the same three predictor-family colors in all importance figures."""
    family = str(feature).split("_", 1)[0]
    return FEATURE_FAMILY_COLORS.get(family, COLOR_REFERENCE)


def add_feature_family_legend(ax, features):
    """Explain the colored bars without repeating individual feature labels."""
    families = {str(feature).split("_", 1)[0] for feature in features}
    names = {"Soil": "Soil properties", "BC": "Biochar properties", "Metal": "Metal properties"}
    handles = [
        mpl.patches.Patch(facecolor=color, edgecolor="none", label=names[family])
        for family, color in FEATURE_FAMILY_COLORS.items() if family in families
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.10),
              ncol=max(1, len(handles)), frameon=False, fontsize=9, borderaxespad=0)


def style_axis(ax, grid_axis='both'):
    """White background, unobtrusive axes, and dark labels for journal figures."""
    ax.set_facecolor("white")
    ax.grid(False, which="both")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_DARK)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", direction="out", length=3.5, width=0.8, colors=COLOR_DARK)


def apply_paper_plot_theme():
    sns.set_theme(style="ticks", context="notebook", palette=PAPER_PALETTE)
    preferred_fonts = ["Arial", "Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
    available_fonts = {font.name for font in mpl.font_manager.fontManager.ttflist}
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font for font in preferred_fonts if font in available_fonts],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.9,
        "axes.edgecolor": COLOR_DARK,
        "axes.labelcolor": COLOR_DARK,
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "text.color": COLOR_DARK,
        "xtick.color": COLOR_DARK,
        "ytick.color": COLOR_DARK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": COLOR_GRID,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "text.usetex": False,
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


apply_paper_plot_theme()

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, KFold, GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import PartialDependenceDisplay
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression

# 数值稳定项：相对误差计算中避免零分母（1个百分点）
EPS_REL = 1.0


def stabilized_relative_error(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    return 100 * np.abs(actual - y_pred) / np.maximum(np.abs(actual), EPS_REL)


def stable_r2(y_true, y_pred):
    if len(y_true) < 2 or np.ptp(y_true) == 0:
        return np.nan
    return r2_score(y_true, y_pred)

st.set_page_config(page_title="Biochar-HM Expert System V3.9.6", page_icon="🌱",
                   layout="wide", initial_sidebar_state="expanded")


def apply_custom_style():
    st.markdown("""
                    <style>
                    .main { background-color: #f7f8fa; }
                    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #3C5488; color: white; border: none; }
                    .stButton>button:hover { background-color: #2C406B; color: white; }
                    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-top: 3px solid #00A087; }
                    h1, h2, h3 { color: #262B33; }
                    .advice-card { background: #ffffff; padding: 16px; border-radius: 8px; border-left: 4px solid #3C5488; margin-bottom: 12px; }
                    .calib-warning { background: #FFF8ED; padding: 12px; border-radius: 6px; border-left: 4px solid #D99023; font-size: 0.95em; }
                    .calib-error { background: #FFF2EF; padding: 12px; border-radius: 6px; border-left: 4px solid #E64B35; font-size: 0.95em; }
                    </style>""", unsafe_allow_html=True)


apply_custom_style()

FEATURE_CONFIG = {
    "numerical": ['Soil_pH', 'Soil_Clay', 'Soil_SOM', 'Soil_CEC', 'Soil_Moisture',
                  'BC_Rate', 'BC_Temp', 'BC_SSA', 'BC_N', 'BC_Ash',
                  'Metal_Total', 'Metal_Available'],
    "categorical": ['BC_Feedstock', 'Metal_Type', 'Soil_Type'],
    "target": 'Immobilization_Efficiency'
}

# 四模型AutoML候选
AUTOML_MODELS = ["XGBoost", "RandomForest", "LightGBM", "SVR"]


class DataGuardian:
    def __init__(self):
        self.constraints = {
            'Soil_pH': (0, 14), 'Soil_Clay': (0, 100), 'Soil_SOM': (0, 500),
            'Soil_CEC': (0, 200), 'Soil_Moisture': (0, 100),
            'BC_Rate': (0, 30), 'BC_Temp': (100, 1200), 'BC_SSA': (0, 3000),
            'BC_N': (0, 30), 'BC_Ash': (0, 100),
            'Metal_Total': (0, 100000), 'Metal_Available': (0, 100000),
            'Immobilization_Efficiency': (0, 100)}

        # CSV values may include an explicit unit suffix in the column's canonical unit.
        self.units = {
            'Soil_pH': '', 'Soil_Clay': '%', 'Soil_SOM': 'g/kg', 'Soil_CEC': 'cmol/kg',
            'Soil_Moisture': '%', 'BC_Rate': '%', 'BC_Temp': '°C', 'BC_SSA': 'm²/g',
            'BC_N': '%', 'BC_Ash': '%', 'Metal_Total': 'mg/kg', 'Metal_Available': 'mg/kg',
            'Immobilization_Efficiency': '%',
        }

    def validate_and_clean(self, df, *, training=True):
        """Validate the shared CSV/form schema and preserve numeric notation."""
        required = FEATURE_CONFIG['numerical'] + FEATURE_CONFIG['categorical']
        target = FEATURE_CONFIG['target']
        if training:
            required = required + [target]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError("缺少必要列：" + "、".join(missing))
        if df.empty:
            raise ValueError("输入数据没有记录。")

        clean = df.copy().reset_index(drop=True)
        logs = []

        def report(col, mask, reason):
            rows = np.flatnonzero(mask.to_numpy(dtype=bool)) + 1
            if len(rows):
                examples = "、".join(map(str, rows[:8]))
                logs.append(f"{col}：{reason}，共 {len(rows)} 条（数据行 {examples}）。")

        bad_rows = pd.Series(False, index=clean.index)
        for col in FEATURE_CONFIG['numerical'] + ([target] if training else []):
            tokens = clean[col].astype('string').str.strip().str.replace('−', '-', regex=False)
            blank = tokens.isna() | tokens.str.lower().isin(['', 'na', 'nan', 'n/a', 'null', 'none', '-'])
            nondetect = tokens.str.match(r'^(?:<|≤|＜)', na=False) | tokens.str.lower().isin(
                ['nd', 'n.d.', 'bdl', '未检出']
            )
            report(col, nondetect, "低于检出限或未检出，按缺失处理")
            tokens = tokens.mask(blank | nondetect)
            unit = self.units[col]
            if unit:
                aliases = {'%': r'[%％]', '°C': r'(?:°C|℃)', 'm²/g': r'm(?:²|\^?2)/g'}
                suffix = aliases.get(unit, re.escape(unit))
                tokens = tokens.str.replace(r'\s*' + suffix + r'$', '', regex=True, flags=re.IGNORECASE)
            grouped = tokens.str.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?', na=False)
            tokens = tokens.where(~grouped, tokens.str.replace(',', '', regex=False))
            values = pd.to_numeric(tokens, errors='coerce').astype(float)
            malformed = tokens.notna() & ~np.isfinite(values)
            report(col, malformed, "数值格式或单位无效")
            values = values.mask(malformed)
            lo, hi = self.constraints[col]
            out = values.notna() & ~values.between(lo, hi)
            report(col, out, f"超出允许范围 [{lo}, {hi}]")
            clean[col] = values
            bad_rows |= malformed | out

        for col in FEATURE_CONFIG['categorical']:
            values = clean[col].astype('string').str.strip()
            empty = values.isna() | values.str.lower().isin(['', 'na', 'nan', 'n/a', 'null', 'none'])
            clean[col] = values.astype(object).mask(empty, np.nan)

        invalid_pair = clean['Metal_Available'] > clean['Metal_Total']
        report('Metal_Available', invalid_pair, "初始有效态浓度大于初始总浓度")
        bad_rows |= invalid_pair
        if training:
            report(target, clean[target].isna(), "目标缺失，无法用于监督训练")
            bad_rows |= clean[target].isna()
        elif bad_rows.any():
            raise ValueError(" ".join(logs))

        clean = clean.loc[~bad_rows].reset_index(drop=True)
        if clean.empty:
            raise ValueError("清洗后没有有效记录。 " + " ".join(logs))
        if training:
            empty_features = [col for col in FEATURE_CONFIG['numerical'] if clean[col].isna().all()]
            if empty_features:
                raise ValueError("以下数值特征没有有效观测，无法拟合插补器：" + "、".join(empty_features))
            logs.append(f"质量管控完成。原始 {len(df)} 条，有效保留 {len(clean)} 条。")
        else:
            missing_values = clean[required].isna().sum()
            missing_values = missing_values[missing_values > 0]
            if not missing_values.empty:
                fields = '、'.join(f'{col} ({count})' for col, count in missing_values.items())
                logs.append(f"以下输入将使用模型拟合的插补规则：{fields}。")
        return clean, logs



class HistoryManager:
    def __init__(self, db_path="biochar_history.db"):
        self.db_path = db_path
        self._init_db()
        self._migrate_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
                                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, soil_ph REAL,
                                metal_type TEXT, bc_rate REAL, bc_temp REAL, predicted_ie REAL,
                                expert_score REAL, feasibility_score REAL)""")

    def _migrate_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()]
            if 'feasibility_score' not in cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN feasibility_score REAL DEFAULT 0")

    def save_record(self, data):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""INSERT INTO predictions
                                (timestamp, soil_ph, metal_type, bc_rate, bc_temp, predicted_ie, expert_score, feasibility_score)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                         (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          data['ph'], data['metal'], data['rate'], data['temp'],
                          data['ie'], data['score'], data.get('feasibility', 0)))

    def get_recent_history(self, limit=100):
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}", conn)

    def get_statistics(self):
        with sqlite3.connect(self.db_path) as conn:
            total = pd.read_sql_query("SELECT COUNT(*) as cnt FROM predictions", conn).iloc[0, 0]
            if total == 0: return None
            s = pd.read_sql_query("""SELECT AVG(predicted_ie) as avg_ie, AVG(bc_rate) as avg_rate,
                                AVG(bc_temp) as avg_temp, AVG(feasibility_score) as avg_feasibility,
                                MAX(predicted_ie) as max_ie, MIN(predicted_ie) as min_ie FROM predictions""",
                                  conn).iloc[0].to_dict()
            s['total'] = total
            return s

    def clear_all(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM predictions")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")


class ScientificPlotter:
    @staticmethod
    def compute_learning_curve(consultant):
        raw = consultant.raw_train_df
        folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(raw))
        smallest = min(len(train_idx) for train_idx, _ in folds)
        sizes = np.unique(np.maximum(2, (np.linspace(0.1, 1.0, 5) * smallest).astype(int)))
        train_scores = [[] for _ in sizes]
        validation_scores = [[] for _ in sizes]
        rng = np.random.default_rng(42)
        prep = BiocharConsultant()
        target = FEATURE_CONFIG['target']
        for train_idx, val_idx in folds:
            val = raw.iloc[val_idx]
            shuffled = rng.permutation(train_idx)
            for i, size in enumerate(sizes):
                train = raw.iloc[shuffled[:size]]
                if train[FEATURE_CONFIG['numerical']].isna().all().any():
                    continue
                X_train, y_train = prep.preprocess(train, is_training=True)
                X_val = prep.preprocess(val, is_training=False)
                model = prep._create_model(consultant.model_name, consultant.best_params)
                model.fit(X_train, y_train)
                train_score = stable_r2(y_train, model.predict(X_train))
                val_score = stable_r2(val[target].to_numpy(), model.predict(X_val))
                if np.isfinite(train_score) and np.isfinite(val_score):
                    train_scores[i].append(train_score)
                    validation_scores[i].append(val_score)
        valid = [i for i, scores in enumerate(validation_scores) if scores]
        if not valid:
            raise ValueError("当前数据无法生成有效学习曲线。")
        return (sizes[valid],
                np.array([np.mean(train_scores[i]) for i in valid]),
                np.array([np.std(train_scores[i]) for i in valid]),
                np.array([np.mean(validation_scores[i]) for i in valid]),
                np.array([np.std(validation_scores[i]) for i in valid]),
                [len(validation_scores[i]) for i in valid])

    @staticmethod
    def plot_learning_curve(train_sizes, tm, ts, em, es, fold_counts):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(train_sizes, tm - ts, tm + ts, alpha=0.13, color=COLOR_ORANGE)
        ax.fill_between(train_sizes, em - es, em + es, alpha=0.13, color=COLOR_BLUE)
        ax.plot(train_sizes, tm, 'o-', color=COLOR_ORANGE, label='Training score',
                markersize=6, linewidth=2.2, markeredgecolor='white', markeredgewidth=0.7)
        ax.plot(train_sizes, em, 's-', color=COLOR_BLUE, label='Cross-validation score',
                markersize=6, linewidth=2.2, markeredgecolor='white', markeredgewidth=0.7)
        ax.set_title('Model Learning Curve (Diagnostic, Supplementary)', pad=12, fontweight='bold')
        ax.set_xlabel('Number of training examples')
        if any(count != 5 for count in fold_counts):
            ax.set_xticks(train_sizes, [f'{size} (n={count})' for size, count in zip(train_sizes, fold_counts)])
        ax.set_ylabel('R² score')
        ax.legend(loc='lower right', frameon=False)
        style_axis(ax)
        fig.tight_layout()
        return fig


class BiocharVisualizer:

    @staticmethod
    def plot_regression_metrics(y_true, y_pred):
        """Independent-test prediction diagnostics in the unified paper palette."""
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

        mn = min(np.nanmin(y_true), np.nanmin(y_pred))
        mx = max(np.nanmax(y_true), np.nanmax(y_pred))
        margin = 0.05 * (mx - mn) if mx > mn else 1.0
        plot_min, plot_max = mn - margin, mx + margin

        ax[0].plot([plot_min, plot_max], [plot_min, plot_max],
                   linestyle='--', linewidth=1.3, color=COLOR_REFERENCE,
                   label='1:1 line', alpha=0.95)
        ax[0].scatter(y_true, y_pred, s=58, alpha=0.82, color=COLOR_BLUE,
                      edgecolors='white', linewidths=0.65, label='Samples')

        if len(y_true) >= 2 and np.nanstd(y_true) > 0:
            slope, intercept = np.polyfit(y_true, y_pred, 1)
            x_fit = np.linspace(plot_min, plot_max, 100)
            ax[0].plot(x_fit, slope * x_fit + intercept, linewidth=2.2,
                       color=COLOR_PINK, label='Linear fit')

        r2 = stable_r2(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        metric_text = f'R² = {r2:.4f}\nRMSE = {rmse:.2f} pp\nMAE = {mae:.2f} pp'
        ax[0].text(0.97, 0.05, metric_text, transform=ax[0].transAxes,
                   ha='right', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             alpha=0.92, edgecolor=COLOR_GRID))
        ax[0].set_xlim(plot_min, plot_max)
        ax[0].set_ylim(plot_min, plot_max)
        ax[0].set_title('Predicted vs. Observed Efficiency', fontweight='bold', pad=10)
        ax[0].set_xlabel('Observed immobilization efficiency (%)')
        ax[0].set_ylabel('Predicted immobilization efficiency (%)')
        ax[0].legend(loc='upper left', frameon=False)
        style_axis(ax[0])

        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True, ax=ax[1], color=COLOR_LIGHT_BLUE,
                     edgecolor='white', linewidth=0.6, alpha=0.85,
                     line_kws={'color': COLOR_BLUE, 'linewidth': 2})
        ax[1].axvline(0, linestyle='--', linewidth=1.8, color=COLOR_ORANGE)
        ax[1].set_title('Residual Error Distribution', fontweight='bold', pad=10)
        ax[1].set_xlabel('Residual (observed - predicted, %)')
        ax[1].set_ylabel('Frequency')
        style_axis(ax[1])
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_feature_importance(model, features, aggregate_onehot=True):
        if not hasattr(model, 'feature_importances_'):
            return None
        importances = model.feature_importances_
        if aggregate_onehot:
            cat_prefixes = ['BC_Feedstock_', 'Metal_Type_', 'Soil_Type_']
            agg, used = {}, set()
            for prefix in cat_prefixes:
                pname = prefix.rstrip('_')
                mask = np.array([f.startswith(prefix) for f in features])
                if mask.any():
                    agg[pname] = importances[mask].sum()
                    used.update(np.where(mask)[0].tolist())
            for i, feat in enumerate(features):
                if i not in used:
                    agg[feat] = importances[i]
            si = sorted(agg.items(), key=lambda x: x[1], reverse=True)
            sf = [x[0] for x in si]
            siv = np.array([x[1] for x in si])
        else:
            idx = np.argsort(importances)[::-1]
            sf = np.array(features)[idx]
            siv = importances[idx]

        fig, ax = plt.subplots(figsize=(10, 6.5))
        bar_colors = [feature_color(f) for f in sf]
        ax.barh(np.arange(len(sf)), siv, color=bar_colors, edgecolor='white', linewidth=0.8, alpha=0.95)
        ax.set_yticks(np.arange(len(sf)))
        ax.set_yticklabels(sf)
        ax.invert_yaxis()
        ax.set_title('Global Feature Importance (15 Original Input Features)', fontweight='bold', pad=12)
        ax.set_xlabel('Relative importance (weight)')
        ax.set_ylabel('')
        for i, v in enumerate(siv):
            ax.text(v + max(siv) * 0.008, i, f'{v:.4f}', va='center', fontsize=8.5, color=COLOR_DARK)
        style_axis(ax, grid_axis='x')
        add_feature_family_legend(ax, sf)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_shap_dependence(shap_values, X_data, feature_name, scaler=None, feature_names=None,
                             interaction_index=None):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        X_plot = X_data.copy()
        if scaler is not None and feature_names is not None:
            num_cols = [c for c in FEATURE_CONFIG['numerical'] if c in feature_names]
            if feature_name in num_cols:
                i = num_cols.index(feature_name)
                X_plot[feature_name] = X_plot[feature_name] * scaler.scale_[i] + scaler.mean_[i]

        shap.dependence_plot(feature_name, shap_values, X_plot,
                             interaction_index=interaction_index, ax=ax, show=False, cmap=SHAP_CMAP)
        # Single-variable dependence plots use the shared deep-blue hue.
        # Interaction plots use the same continuous map as the beeswarm.
        if interaction_index is None:
            for coll in ax.collections:
                try:
                    coll.set_facecolor(COLOR_BLUE)
                    coll.set_edgecolor('white')
                    coll.set_alpha(0.82)
                except Exception:
                    pass
        ax.axhline(0, color=COLOR_REFERENCE, linestyle='--', linewidth=1.2, alpha=0.75)
        ax.set_title(f'SHAP Dependence Plot: {feature_name}', fontweight='bold', pad=10)
        style_axis(ax)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_pdp(model, X, features_to_plot, scaler=None, feature_names=None):
        nf = len(features_to_plot)
        fig, ax = plt.subplots(1, nf, figsize=(6.5 * nf, 5.5))
        if nf == 1:
            ax = np.array([ax])
        else:
            ax = np.ravel(ax)
        PartialDependenceDisplay.from_estimator(
            estimator=model, X=X, features=features_to_plot, grid_resolution=50, ax=ax,
            line_kw={'color': COLOR_BLUE, 'linewidth': 2.8})
        ncp = [c for c in FEATURE_CONFIG['numerical'] if c in feature_names] if feature_names else []
        for i, a in enumerate(ax):
            fn = features_to_plot[i]
            if scaler is not None and fn in ncp:
                idx = ncp.index(fn)
                mv, sv = scaler.mean_[idx], scaler.scale_[idx]

                def mf(m, s):
                    def f(x, pos):
                        rv = x * s + m
                        return f'{rv:.0f}' if abs(rv) >= 10 else f'{rv:.1f}'

                    return ticker.FuncFormatter(f)

                a.xaxis.set_major_formatter(mf(mv, sv))
            a.axhline(a.get_ylim()[0], color='none')
            style_axis(a)
            for line in a.lines:
                line.set_color(COLOR_BLUE)
        fig.suptitle('Partial Dependence Plot (Non-linear Response Analysis)',
                     fontweight='bold', fontsize=14, y=1.03)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_model_comparison_bar(metrics_df):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        models = metrics_df['Model'].dropna().unique().tolist()
        palette = {m: MODEL_COLOR_MAP.get(m, PAPER_PALETTE[i % len(PAPER_PALETTE)])
                   for i, m in enumerate(models)}
        sns.barplot(data=metrics_df, x='Metric', y='Value', hue='Model', palette=palette,
                    ax=ax, edgecolor='white', linewidth=0.8, alpha=0.95)
        ax.set_xticklabels(['R²', 'RMSE', 'MAE'], fontsize=12, fontweight='bold')
        ax.set_title('Baseline Model Comparison (Supplementary)', fontweight='bold', fontsize=12, pad=12)
        ax.set_xlabel('')
        ax.set_ylabel('Score / error value', fontsize=12)
        mv = metrics_df['Value'].max()
        lower = min(0, metrics_df['Value'].min())
        ax.set_ylim(lower * 1.15, max(mv * 1.25, 0.01))
        ax.axhline(0, color=COLOR_GRID, linewidth=0.8)
        ax.legend(title='Models', loc='upper right', frameon=False)
        for p in ax.patches:
            h = p.get_height()
            if np.isfinite(h) and h != 0:
                ax.annotate(f'{h:.2f}', (p.get_x() + p.get_width() / 2., h), ha='center', va='bottom',
                            fontsize=8.5, color=COLOR_DARK, xytext=(0, 4 if h > 0 else -9), textcoords='offset points')
        style_axis(ax, grid_axis='y')
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_taylor_diagram(models_stats, ref_std):
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection='polar')
        finite_stats = [item for item in models_stats if np.isfinite(item['corr'])]
        show_negative = any(item['corr'] < 0 for item in finite_stats)
        corrs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
        if show_negative:
            corrs = np.r_[-1, -0.8, -0.6, -0.4, -0.2, corrs]
        angles = np.arccos(corrs)
        theta_max = np.pi if show_negative else np.pi / 2
        ax.set_thetamax(np.degrees(theta_max))
        ax.set_xticks(angles);
        ax.set_xticklabels([str(c) for c in corrs], fontsize=10)
        ax.set_xlabel('Standard deviation', fontsize=12, labelpad=20, fontweight='bold')
        ax.text(np.pi / 4, ax.get_rmax() * 1.20, 'Correlation coefficient', rotation=-45,
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax.plot(0, ref_std, marker='*', color=COLOR_DARK, markersize=18, label='Observation', linestyle='None')
        radial_limit = max([float(ref_std)] + [float(item['std']) for item in finite_stats]) * 1.1
        radial_limit = max(radial_limit, 0.01)
        ax.set_rmax(radial_limit)
        rs, ts = np.meshgrid(np.linspace(0, radial_limit, 100), np.linspace(0, theta_max, 100))
        rms = np.sqrt(ref_std ** 2 + rs ** 2 - 2 * ref_std * rs * np.cos(ts))
        contours = ax.contour(ts, rs, rms, levels=5, colors=COLOR_GRID, alpha=0.95, linestyles='--')
        ax.clabel(contours, inline=True, fontsize=9, fmt='%.1f')
        markers = ['o', 's', '^', 'D', 'P']
        for i, stat in enumerate(finite_stats):
            theta = np.arccos(stat['corr'])
            radius = stat['std']
            c = MODEL_COLOR_MAP.get(stat['name'], PAPER_PALETTE[i % len(PAPER_PALETTE)])
            ax.plot(theta, radius, marker=markers[i % len(markers)], color=c, markersize=11,
                    markeredgecolor='white', markeredgewidth=0.8, linestyle='None',
                    label=stat['name'], alpha=0.95)
        if len(finite_stats) < len(models_stats):
            ax.text(0, -0.10, 'Models with undefined correlation are omitted.',
                    transform=ax.transAxes, fontsize=8)
        ax.legend(title='Models', loc='upper right', bbox_to_anchor=(0.98, 0.98), frameon=False)
        ax.set_title('Taylor Diagram (Baseline Models, Supplementary)', fontsize=13, fontweight='bold', y=1.08)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_feasibility_calibration(df, weights, thresholds, metrics, separable=True):
        df = df.copy()
        df['total_score'] = (weights['boundary'] * df['boundary_score'] +
                             weights['deviation'] * df['deviation_score'] +
                             weights['distance'] * df['distance_score'])
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        sns.regplot(data=df, x='total_score', y='rel_error', ax=ax[0],
                    scatter_kws={'alpha': 0.72, 'color': COLOR_BLUE, 'edgecolor': 'white', 's': 42},
                    line_kws={'color': COLOR_PINK, 'lw': 2.0, 'label': 'Linear fit'})
        if separable:
            ax[0].axvline(thresholds['high'], color=COLOR_OLIVE, linestyle='--', lw=2, label='High threshold')
            ax[0].axvline(thresholds['medium'], color=COLOR_ORANGE, linestyle=':', lw=2, label='Medium threshold')
        textstr = f"Spearman ρ = {metrics['spearman_correlation']:.3f}\n(exploratory, not primary evidence)"
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.96, edgecolor=COLOR_GRID)
        ax[0].text(0.95, 0.95, textstr, transform=ax[0].transAxes, fontsize=10, fontweight='bold',
                   verticalalignment='top', horizontalalignment='right', bbox=props)
        ax[0].set_title('Feasibility Score vs. OOF Relative Error', fontweight='bold', pad=10)
        ax[0].set_xlabel('Comprehensive feasibility score')
        ax[0].set_ylabel('Relative error (%)')
        ax[0].legend(loc='upper right', bbox_to_anchor=(0.95, 0.82), framealpha=0.9)
        ax[0].set_ylim(-2, min(60, df['rel_error'].max() * 1.1))
        style_axis(ax[0])

        if separable:
            categories = ['High\nFeasibility', 'Medium\nFeasibility', 'Low\nFeasibility (Risk)']
            mapes = [metrics['high_level_mape'], metrics['mid_level_mape'], metrics['low_level_mape']]
            counts = [metrics['high_level_count'], metrics['mid_level_count'], metrics['low_level_count']]
            colors = [COLOR_OLIVE, COLOR_ORANGE, COLOR_PINK]
            sns.barplot(x=categories, y=mapes, ax=ax[1], palette=colors,
                        edgecolor='white', linewidth=0.8, alpha=0.95)
            ax[1].set_title('Average Error by Feasibility Level', fontweight='bold', pad=10)
            ax[1].set_ylabel('Mean relative error (%)')
            ax[1].set_ylim(0, max(mapes) * 1.25)
            for i, p in enumerate(ax[1].patches):
                h = p.get_height()
                ax[1].annotate(f'{h:.1f}%', (p.get_x() + p.get_width() / 2., h), ha='center', va='bottom',
                               fontsize=10, fontweight='bold', xytext=(0, 4), textcoords='offset points')
                ax[1].annotate(f'n={counts[i]}', (p.get_x() + p.get_width() / 2., h / 2), ha='center', va='center',
                               fontsize=9, color=COLOR_DARK, fontweight='normal')
            style_axis(ax[1], grid_axis='y')
        else:
            ax[1].text(0.5, 0.5, 'Three-level stratification\nnot established\n(T_H ≤ T_M)',
                       transform=ax[1].transAxes, ha='center', va='center', fontsize=14,
                       fontweight='normal', color=COLOR_DARK)
            ax[1].set_title('Feasibility Stratification: Not Established', fontweight='bold', pad=10)
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            for sp in ax[1].spines.values(): sp.set_visible(False)
        fig.tight_layout()
        return fig


# =================================================================
# 功能
# =================================================================
class ScenarioResponseAnalyzer:
    TARGET = FEATURE_CONFIG['target']

    DESIGN_VARS = [
        'BC_Rate',
        'BC_Temp',
        'BC_N',
        'BC_SSA'
    ]

    PH_ORDER = [
        'Acidic (<6.5)',
        'Neutral (6.5-7.5)',
        'Alkaline (>=7.5)'
    ]

    METAL_PREFERRED_ORDER = [
        'Cd', 'Pb', 'Cu', 'Zn', 'As',
        'Cr', 'Mn', 'Ni', 'Hg'
    ]

    # -------------------------------------------------------------
    # 1. 构建3.4初始情景数据
    # -------------------------------------------------------------
    @staticmethod
    def prepare_scenario_dataframe(df):
        """
        使用完整清洗数据库构建描述性分析数据。

        新增：
        - Soil_pH_Group
        - R_bio = Metal_Available / Metal_Total
        - R_bio_Group

        R_bio仅用于初始情景描述，
        不参与机器学习模型训练。
        """

        data = df.copy()

        # 确保关键数值列为numeric
        numeric_needed = [
            'Soil_pH',
            'Metal_Total',
            'Metal_Available',
            FEATURE_CONFIG['target']
        ]

        for col in numeric_needed:
            if col in data.columns:
                data[col] = pd.to_numeric(
                    data[col],
                    errors='coerce'
                )

        # =========================================================
        # Soil pH分组
        # pH < 6.5
        # 6.5 <= pH < 7.5
        # pH >= 7.5
        # =========================================================
        data['Soil_pH_Group'] = pd.cut(
            data['Soil_pH'],
            bins=[-np.inf, 6.5, 7.5, np.inf],
            labels=ScenarioResponseAnalyzer.PH_ORDER,
            right=False
        )

        # =========================================================
        # 初始有效态比例
        # R_bio = C_bio / C_0
        # 仅用于描述性场景分析，不是模型输入变量
        # =========================================================
        valid_total = (
                data['Metal_Total'].notna()
                & data['Metal_Available'].notna()
                & (data['Metal_Total'] > 0)
        )

        data['R_bio'] = np.nan

        data.loc[valid_total, 'R_bio'] = (
                data.loc[valid_total, 'Metal_Available']
                / data.loc[valid_total, 'Metal_Total']
        )

        # 理论上DataGuardian已保证C_bio <= C_0
        # 这里再做一次数值保护
        data.loc[
            (data['R_bio'] < 0)
            | (data['R_bio'] > 1),
            'R_bio'
        ] = np.nan

        data['R_bio_pct'] = 100 * data['R_bio']

        # =========================================================
        # R_bio按数据库三分位数划分Low / Medium / High
        # 不人为设置固定环境阈值
        # =========================================================
        valid_rbio = data['R_bio'].dropna()

        rbio_info = {
            'q33': np.nan,
            'q67': np.nan,
            'n_valid': len(valid_rbio)
        }

        data['R_bio_Group'] = pd.Series(None, index=data.index, dtype=object)

        if len(valid_rbio) >= 6:

            q33 = valid_rbio.quantile(1 / 3)
            q67 = valid_rbio.quantile(2 / 3)

            rbio_info['q33'] = float(q33)
            rbio_info['q67'] = float(q67)

            # 如果分位点可以正常区分
            if q33 < q67:

                conditions = [
                    data['R_bio'] <= q33,
                    (
                            (data['R_bio'] > q33)
                            & (data['R_bio'] <= q67)
                    ),
                    data['R_bio'] > q67
                ]

                choices = [
                    'Low',
                    'Medium',
                    'High'
                ]

                data['R_bio_Group'] = np.select(
                    conditions, np.asarray(choices, dtype=object), default=None
                )

            # 若大量重复值导致两个分位点相同
            # 使用百分位秩进行稳健分组
            else:

                pct_rank = data['R_bio'].rank(
                    method='average',
                    pct=True
                )

                data.loc[
                    pct_rank <= 1 / 3,
                    'R_bio_Group'
                ] = 'Low'

                data.loc[
                    (pct_rank > 1 / 3)
                    & (pct_rank <= 2 / 3),
                    'R_bio_Group'
                ] = 'Medium'

                data.loc[
                    pct_rank > 2 / 3,
                    'R_bio_Group'
                ] = 'High'

        return data, rbio_info

    # -------------------------------------------------------------
    # 2. 获取Metal顺序
    # -------------------------------------------------------------
    @staticmethod
    def get_metal_order(data):

        if 'Metal_Type' not in data.columns:
            return []

        present = (
            data['Metal_Type']
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        preferred = [
            x
            for x in ScenarioResponseAnalyzer.METAL_PREFERRED_ORDER
            if x in present
        ]

        remaining = sorted([
            x
            for x in present
            if x not in preferred
        ])

        return preferred + remaining

    # -------------------------------------------------------------
    # 3. 描述性统计
    # -------------------------------------------------------------
    @staticmethod
    def descriptive_statistics(
            data,
            group_col,
            group_order=None
    ):

        target = ScenarioResponseAnalyzer.TARGET

        work = data[
            [group_col, target]
        ].dropna().copy()

        if work.empty:
            return pd.DataFrame()

        rows = []

        if group_order is None:
            groups = list(
                work[group_col]
                .astype(str)
                .unique()
            )
        else:
            groups = group_order

        for group in groups:

            vals = work.loc[
                work[group_col].astype(str) == str(group),
                target
            ].dropna()

            if len(vals) == 0:
                continue

            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)

            rows.append({
                'Group': str(group),
                'N': int(len(vals)),
                'Mean_EHM': float(vals.mean()),
                'SD_EHM': float(vals.std(ddof=1))
                if len(vals) > 1 else np.nan,
                'Median_EHM': float(vals.median()),
                'Q1_EHM': float(q1),
                'Q3_EHM': float(q3),
                'IQR_EHM': float(q3 - q1),
                'Min_EHM': float(vals.min()),
                'Max_EHM': float(vals.max())
            })

        return pd.DataFrame(rows)

    # -------------------------------------------------------------
    # 4. 探索性Kruskal-Wallis
    # -------------------------------------------------------------
    @staticmethod
    def kruskal_test(
            data,
            group_col
    ):
        """
        仅作为探索性组间比较。
        文献数据库不是随机平衡设计，
        因此P值不能解释为单一因素的因果效应。
        """

        target = ScenarioResponseAnalyzer.TARGET

        groups = []

        labels = []

        for label, sub in data.groupby(
                group_col,
                observed=True
        ):

            vals = sub[target].dropna().values

            if len(vals) >= 3:
                groups.append(vals)
                labels.append(str(label))

        if len(groups) < 2:
            return {
                'H': np.nan,
                'p': np.nan,
                'n_groups': len(groups)
            }

        try:
            result = kruskal(*groups)

            return {
                'H': float(result.statistic),
                'p': float(result.pvalue),
                'n_groups': len(groups)
            }

        except Exception:

            return {
                'H': np.nan,
                'p': np.nan,
                'n_groups': len(groups)
            }

    # -------------------------------------------------------------
    # 5. 将SHAP重新聚合回15个原始变量
    # -------------------------------------------------------------
    @staticmethod
    def build_sample_level_grouped_shap(
            shap_values,
            feature_names
    ):
        """
        将One-Hot后的SHAP重新聚合为原始15变量。

        连续变量：
            abs(phi_j)

        分类变量：
            abs(sum(phi_dummy))

        与论文Methods中的grouped SHAP定义保持一致。
        """

        shap_array = np.asarray(shap_values)

        if shap_array.ndim != 2:
            raise ValueError(
                "SHAP values必须为二维数组 "
                "(n_samples, n_features)。"
            )

        feature_names = list(feature_names)

        if shap_array.shape[1] != len(feature_names):
            raise ValueError(
                "SHAP列数与feature_names长度不一致。"
            )

        output = {}

        # =========================================================
        # 连续变量
        # =========================================================
        for feature in FEATURE_CONFIG['numerical']:

            if feature in feature_names:
                idx = feature_names.index(feature)

                output[feature] = np.abs(
                    shap_array[:, idx]
                )

        # =========================================================
        # 分类变量
        # 样本内signed aggregation后再取absolute
        # =========================================================
        for feature in FEATURE_CONFIG['categorical']:

            prefix = feature + '_'

            cols = [
                i
                for i, name in enumerate(feature_names)
                if name.startswith(prefix)
            ]

            if len(cols) > 0:
                output[feature] = np.abs(
                    shap_array[:, cols].sum(axis=1)
                )

        return pd.DataFrame(output)

    # -------------------------------------------------------------
    # 6. subgroup normalized SHAP
    # -------------------------------------------------------------
    @staticmethod
    def build_subgroup_shap(
            raw_train_df,
            grouped_sample_shap,
            min_group_n=10,
            top_n_features=10
    ):
        """
        构建：
        - Metal_Type subgroup
        - Soil pH subgroup

        每个场景内部归一化：
            I* = 100 * I_jk / sum_j(I_jk)

        返回：
        heatmap_df
        summary_df
        """

        raw = raw_train_df.reset_index(
            drop=True
        ).copy()

        shap_df = grouped_sample_shap.reset_index(
            drop=True
        ).copy()

        if len(raw) != len(shap_df):
            raise ValueError(
                "raw_train_df与SHAP样本数量不一致。"
            )

        # =========================================================
        # pH组
        # =========================================================
        raw['Soil_pH_Group'] = pd.cut(
            pd.to_numeric(
                raw['Soil_pH'],
                errors='coerce'
            ),
            bins=[-np.inf, 6.5, 7.5, np.inf],
            labels=ScenarioResponseAnalyzer.PH_ORDER,
            right=False
        )

        subgroup_rows = []
        summary_rows = []

        # ---------------------------------------------------------
        # 内部函数：加入一个subgroup
        # ---------------------------------------------------------
        def add_group(
                label,
                mask,
                group_type
        ):

            mask = np.asarray(mask)

            n = int(mask.sum())

            if n < min_group_n:
                return

            mean_imp = (
                shap_df.loc[mask]
                .mean(axis=0)
            )

            total = mean_imp.sum()

            if (
                    total is None
                    or not np.isfinite(total)
                    or total <= 0
            ):
                return

            normalized = (
                    100.0
                    * mean_imp
                    / total
            )

            row = normalized.to_dict()

            row['Scenario'] = label
            row['Scenario_Type'] = group_type
            row['N'] = n

            subgroup_rows.append(row)

            # Top 3总体变量
            top3 = (
                normalized
                .sort_values(
                    ascending=False
                )
                .head(3)
            )

            # Top design variable
            design_existing = [
                f
                for f in ScenarioResponseAnalyzer.DESIGN_VARS
                if f in normalized.index
            ]

            if len(design_existing) > 0:

                design_scores = normalized[
                    design_existing
                ].sort_values(
                    ascending=False
                )

                top_design = (
                    design_scores.index[0]
                )

                top_design_share = float(
                    design_scores.iloc[0]
                )

            else:

                top_design = None
                top_design_share = np.nan

            summary_rows.append({
                'Scenario': label,
                'Scenario_Type': group_type,
                'N': n,
                'Top_1': top3.index[0]
                if len(top3) > 0 else None,
                'Top_2': top3.index[1]
                if len(top3) > 1 else None,
                'Top_3': top3.index[2]
                if len(top3) > 2 else None,
                'Top_design_variable': top_design,
                'Top_design_SHAP_%':
                    top_design_share
            })

        # =========================================================
        # Metal_Type分组
        # =========================================================
        if 'Metal_Type' in raw.columns:

            metal_order = (
                ScenarioResponseAnalyzer
                .get_metal_order(raw)
            )

            for metal in metal_order:
                mask = (
                        raw['Metal_Type']
                        .astype(str)
                        == str(metal)
                )

                add_group(
                    label=f'Metal: {metal}',
                    mask=mask,
                    group_type='Metal_Type'
                )

        # =========================================================
        # pH分组
        # =========================================================
        for ph_group in ScenarioResponseAnalyzer.PH_ORDER:
            mask = (
                    raw['Soil_pH_Group']
                    .astype(str)
                    == str(ph_group)
            )

            add_group(
                label=f'pH: {ph_group}',
                mask=mask,
                group_type='Soil_pH'
            )

        subgroup_df = pd.DataFrame(
            subgroup_rows
        )

        summary_df = pd.DataFrame(
            summary_rows
        )

        if subgroup_df.empty:
            return (
                pd.DataFrame(),
                summary_df
            )

        # =========================================================
        # 选全局平均贡献最高的Top-N变量
        # =========================================================
        feature_cols = [
            c
            for c in shap_df.columns
            if c in subgroup_df.columns
        ]

        overall_rank = (
            shap_df[feature_cols]
            .mean(axis=0)
            .sort_values(
                ascending=False
            )
        )

        top_features = (
            overall_rank
            .head(top_n_features)
            .index
            .tolist()
        )

        heatmap_df = (
            subgroup_df
            .set_index('Scenario')[
                top_features
            ]
        )

        return (
            heatmap_df,
            summary_df
        )

    # -------------------------------------------------------------
    # 7. 给箱线图标签增加样本量
    # -------------------------------------------------------------
    @staticmethod
    def labels_with_n(
            data,
            group_col,
            order
    ):

        target = ScenarioResponseAnalyzer.TARGET

        labels = []

        for item in order:
            n = data.loc[
                data[group_col]
                .astype(str)
                == str(item),
                target
            ].notna().sum()

            labels.append(
                f'{item}\n(n={n})'
            )

        return labels

    # =============================================================
    # -------------------------------------------------------------
    # Publication axis style
    # -------------------------------------------------------------
    @staticmethod
    def _style_publication_axis(
            ax,
            xlabel=None,
            ylabel=None,
            ylim=None,
            grid_axis='y'
    ):
        """
        统一Biochar投稿图风格。
        """

        if xlabel is not None:
            ax.set_xlabel(
                xlabel,
                fontsize=12,
                fontweight='normal',
                labelpad=8
            )

        if ylabel is not None:
            ax.set_ylabel(
                ylabel,
                fontsize=12,
                fontweight='normal',
                labelpad=8
            )

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=10,
            width=1.1,
            length=4
        )

        style_axis(ax, grid_axis=grid_axis)

    # -------------------------------------------------------------
    # 标准文件名
    # -------------------------------------------------------------
    @staticmethod
    def standard_figure_filename(
            fig_no,
            subtag,
            title_key,
            dpi,
            ext
    ):
        """
        Example:
        Fig6a_MetalType_Response_600dpi.png
        """

        safe_title = (
            str(title_key)
            .replace(" ", "")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "pct")
        )

        return (
            f"Fig{fig_no}{subtag}_"
            f"{safe_title}_"
            f"{dpi}dpi.{ext}"
        )

    # -------------------------------------------------------------
    # Figure -> PNG
    # -------------------------------------------------------------
    @staticmethod
    def figure_to_png(
            fig,
            dpi=600
    ):
        buffer = io.BytesIO()

        fig.savefig(
            buffer,
            format='png',
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.08
        )

        buffer.seek(0)

        return buffer.getvalue()

    # -------------------------------------------------------------
    # Figure -> TIFF
    # -------------------------------------------------------------
    @staticmethod
    def figure_to_tiff(
            fig,
            dpi=600
    ):
        """
        优先使用LZW压缩TIFF。
        若当前Pillow/Matplotlib不支持，则自动退回普通TIFF。
        """

        buffer = io.BytesIO()

        try:
            fig.savefig(
                buffer,
                format='tiff',
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.08,
                pil_kwargs={
                    'compression': 'tiff_lzw'
                }
            )

        except Exception:
            buffer = io.BytesIO()

            fig.savefig(
                buffer,
                format='tiff',
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.08
            )

        buffer.seek(0)

        return buffer.getvalue()

    # -------------------------------------------------------------
    # PNG + TIFF下载
    # -------------------------------------------------------------
    @staticmethod
    def render_download_buttons(
            fig,
            fig_no,
            subtag,
            title_key,
            dpi=600,
            key_prefix="",
            cache=None,
            options=None
    ):
        """
        Streamlit中同时输出PNG/TIFF下载按钮。
        """

        cache_key = (dpi, options)
        cached = cache.get(key_prefix) if cache is not None else None
        if cached is not None and cached[0] == cache_key:
            _, png_data, tiff_data = cached
        else:
            png_data = ScenarioResponseAnalyzer.figure_to_png(fig, dpi=dpi)
            tiff_data = ScenarioResponseAnalyzer.figure_to_tiff(fig, dpi=dpi)
            if cache is not None:
                cache[key_prefix] = (cache_key, png_data, tiff_data)

        png_name = (
            ScenarioResponseAnalyzer
            .standard_figure_filename(
                fig_no=fig_no,
                subtag=subtag,
                title_key=title_key,
                dpi=dpi,
                ext='png'
            )
        )

        tiff_name = (
            ScenarioResponseAnalyzer
            .standard_figure_filename(
                fig_no=fig_no,
                subtag=subtag,
                title_key=title_key,
                dpi=dpi,
                ext='tiff'
            )
        )

        col_png, col_tiff = st.columns(2)

        with col_png:
            st.download_button(
                label=f"⬇️ PNG ({dpi} dpi)",
                data=png_data,
                file_name=png_name,
                mime='image/png',
                key=f"{key_prefix}_png_{dpi}"
            )

        with col_tiff:
            st.download_button(
                label=f"⬇️ TIFF ({dpi} dpi)",
                data=tiff_data,
                file_name=tiff_name,
                mime='image/tiff',
                key=f"{key_prefix}_tiff_{dpi}"
            )

    # =============================================================
    # Fig.
    # =============================================================
    @staticmethod
    def plot_figure6a_metal(
            scenario_data
    ):

        target = ScenarioResponseAnalyzer.TARGET

        fig, ax = plt.subplots(
            figsize=(6.6, 5.2)
        )

        metal_order = (
            ScenarioResponseAnalyzer
            .get_metal_order(
                scenario_data
            )
        )

        if len(metal_order) > 0:

            palette = make_palette_dict(
                metal_order
            )

            sns.boxplot(
                data=scenario_data,
                x='Metal_Type',
                y=target,
                order=metal_order,
                palette=palette,
                saturation=0.90,
                ax=ax,
                width=0.62,
                linewidth=1.2,
                showfliers=True,
                fliersize=2.8,
            boxprops={'edgecolor': COLOR_DARK, 'linewidth': 0.9},
            medianprops={'color': COLOR_DARK, 'linewidth': 1.5},
            whiskerprops={'color': COLOR_DARK, 'linewidth': 0.9},
            capprops={'color': COLOR_DARK, 'linewidth': 0.9},
            flierprops={'marker': 'o', 'markerfacecolor': 'none',
                        'markeredgecolor': COLOR_REFERENCE, 'markeredgewidth': 0.6}
            )

            labels = (
                ScenarioResponseAnalyzer
                .labels_with_n(
                    scenario_data,
                    'Metal_Type',
                    metal_order
                )
            )

            ax.set_xticklabels(
                labels,
                rotation=0
            )

        else:

            ax.text(
                0.5,
                0.5,
                'Insufficient metal-type data',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=11
            )

        ax.set_title(
            '(a) Immobilization efficiency by metal type',
            fontsize=13,
            fontweight='bold',
            loc='left',
            pad=10
        )

        # 所有EHM分图统一0–100%
        ScenarioResponseAnalyzer._style_publication_axis(
            ax,
            xlabel='Metal type',
            ylabel='Immobilization efficiency (%)',
            ylim=(0, 100),
            grid_axis='y'
        )

        fig.tight_layout()

        return fig

    # =============================================================
    # Fig
    # =============================================================
    @staticmethod
    def plot_figure6b_ph(
            scenario_data
    ):

        target = ScenarioResponseAnalyzer.TARGET

        fig, ax = plt.subplots(
            figsize=(6.6, 5.2)
        )

        sns.boxplot(
            data=scenario_data,
            x='Soil_pH_Group',
            y=target,
            order=ScenarioResponseAnalyzer.PH_ORDER,
            palette=PH_COLOR_MAP,
            saturation=0.90,
            ax=ax,
            width=0.62,
            linewidth=1.2,
            showfliers=True,
            fliersize=2.8,
            boxprops={'edgecolor': COLOR_DARK, 'linewidth': 0.9},
            medianprops={'color': COLOR_DARK, 'linewidth': 1.5},
            whiskerprops={'color': COLOR_DARK, 'linewidth': 0.9},
            capprops={'color': COLOR_DARK, 'linewidth': 0.9},
            flierprops={'marker': 'o', 'markerfacecolor': 'none',
                        'markeredgecolor': COLOR_REFERENCE, 'markeredgewidth': 0.6}
        )

        ph_labels = (
            ScenarioResponseAnalyzer
            .labels_with_n(
                scenario_data,
                'Soil_pH_Group',
                ScenarioResponseAnalyzer.PH_ORDER
            )
        )

        ax.set_xticklabels(
            ph_labels,
            rotation=0
        )

        ax.set_title(
            '(b) Immobilization efficiency by soil pH',
            fontsize=13,
            fontweight='bold',
            loc='left',
            pad=10
        )

        ScenarioResponseAnalyzer._style_publication_axis(
            ax,
            xlabel='Initial soil pH group',
            ylabel='Immobilization efficiency (%)',
            ylim=(0, 100),
            grid_axis='y'
        )

        fig.tight_layout()

        return fig

    # =============================================================
    # Fig
    # =============================================================
    @staticmethod
    def plot_figure6c_rbio(
            scenario_data
    ):

        target = ScenarioResponseAnalyzer.TARGET

        fig, ax = plt.subplots(
            figsize=(6.6, 5.2)
        )

        rbio_order = [
            'Low',
            'Medium',
            'High'
        ]

        rbio_plot = (
            scenario_data[
                scenario_data[
                    'R_bio_Group'
                ].notna()
            ].copy()
        )

        if not rbio_plot.empty:

            sns.boxplot(
                data=rbio_plot,
                x='R_bio_Group',
                y=target,
                order=rbio_order,
                palette=RBIO_COLOR_MAP,
                saturation=1.0,
                ax=ax,
                width=0.62,
                linewidth=1.2,
                showfliers=True,
                fliersize=2.8,
            boxprops={'edgecolor': COLOR_DARK, 'linewidth': 0.9},
            medianprops={'color': COLOR_DARK, 'linewidth': 1.5},
            whiskerprops={'color': COLOR_DARK, 'linewidth': 0.9},
            capprops={'color': COLOR_DARK, 'linewidth': 0.9},
            flierprops={'marker': 'o', 'markerfacecolor': 'none',
                        'markeredgecolor': COLOR_REFERENCE, 'markeredgewidth': 0.6}
            )

            labels = (
                ScenarioResponseAnalyzer
                .labels_with_n(
                    rbio_plot,
                    'R_bio_Group',
                    rbio_order
                )
            )

            ax.set_xticklabels(
                labels,
                rotation=0
            )

        else:

            ax.text(
                0.5,
                0.5,
                'Insufficient C_bio/C_0 data',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=11
            )

        ax.set_title(
            '(c) Response across initial bioavailable fractions',
            fontsize=13,
            fontweight='bold',
            loc='left',
            pad=10
        )

        ScenarioResponseAnalyzer._style_publication_axis(
            ax,
            xlabel='Initial bioavailable fraction group',
            ylabel='Immobilization efficiency (%)',
            ylim=(0, 100),
            grid_axis='y'
        )

        fig.tight_layout()

        return fig

    # =============================================================
    # Fig
    # =============================================================
    @staticmethod
    def plot_figure6d_heatmap(
            heatmap_df
    ):

        if (
                heatmap_df is not None
                and not heatmap_df.empty
        ):

            n_rows, n_cols = (
                heatmap_df.shape
            )

            # 自动根据行列数量调整
            fig_w = max(
                7.5,
                0.85 * n_cols + 2.8
            )

            fig_h = max(
                5.2,
                0.55 * n_rows + 2.2
            )

        else:

            fig_w = 7.5
            fig_h = 5.2

        fig, ax = plt.subplots(
            figsize=(fig_w, fig_h)
        )

        if (
                heatmap_df is not None
                and not heatmap_df.empty
        ):

            ax.set_facecolor(COLOR_GRID)
            hm = sns.heatmap(
                heatmap_df,
                annot=True,
                fmt='.1f',
                cmap=HEATMAP_CMAP,
                vmin=0,
                linewidths=0.65,
                linecolor='white',
                annot_kws={
                    'fontsize': 9
                },
                cbar_kws={
                    'label':
                        'Normalized SHAP contribution (%)',
                    'shrink': 0.88,
                    'pad': 0.02
                },
                ax=ax
            )

            ax.set_xlabel(
                'Predictor',
                fontsize=12,
                labelpad=8
            )

            ax.set_ylabel(
                'Scenario',
                fontsize=12,
                labelpad=8
            )

            ax.tick_params(
                axis='x',
                labelsize=9,
                rotation=45
            )

            ax.tick_params(
                axis='y',
                labelsize=9,
                rotation=0
            )

            # Match annotation contrast to the rendered cell background.
            values = heatmap_df.to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            for annotation, value in zip(hm.texts, finite_values):
                red, green, blue, _ = HEATMAP_CMAP(hm.collections[0].norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                annotation.set_color("white" if luminance < 0.53 else COLOR_DARK)

            # Colorbar字体统一
            cbar = hm.collections[0].colorbar

            cbar.ax.tick_params(
                labelsize=9
            )

            cbar.ax.yaxis.label.set_size(
                10
            )

        else:

            ax.text(
                0.5,
                0.5,
                'Insufficient subgroup data',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=11
            )

            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(
            '(d) Scenario-specific normalized SHAP contributions',
            fontsize=13,
            fontweight='bold',
            loc='left',
            pad=10
        )

        fig.tight_layout()

        return fig


class AdvancedOptimizer:
    def __init__(self, consultant):
        self.consultant = consultant

    def run_optimization(self, soil_params, weights):
        features = ['BC_Rate', 'BC_Temp', 'BC_N', 'BC_SSA']
        hardware_limits = [(0.5, 10.0), (300, 800), (0.1, 15.0), (20, 1000)]
        ws = sum(weights[key] for key in ('ie', 'cost', 'fert'))
        wie, wco, wfe = (
            tuple(weights[key] / ws for key in ('ie', 'cost', 'fert'))
            if ws > 0 else (1.0, 0.0, 0.0)
        )

        # Keep complete, observed design combinations inside the engineering limits.
        observed = self.consultant.raw_train_df.dropna(subset=features)
        for feature, (lo, hi) in zip(features, hardware_limits):
            observed = observed[observed[feature].between(lo, hi)]
        same_feedstock = observed[observed['BC_Feedstock'] == soil_params['BC_Feedstock']]
        same_metal = observed[observed['Metal_Type'] == soil_params['Metal_Type']]
        same_both = same_feedstock[same_feedstock['Metal_Type'] == soil_params['Metal_Type']]
        references = [
            (same_both, "同原料、同污染物"),
            (same_feedstock, "同原料"),
            (same_metal, "同污染物"),
            (observed, "全部训练样本"),
        ]
        if observed.empty:
            raise ValueError("工程范围内没有四项配方参数均完整的训练记录。")
        design, scope = next(
            ((rows, label) for rows, label in references if len(rows[features].drop_duplicates()) >= 5),
            (observed, "全部训练样本"),
        )
        anchors = design[features].drop_duplicates().to_numpy()
        lo, hi = anchors.min(axis=0), anchors.max(axis=0)
        bounds = list(zip(lo, hi))
        scale = np.where(hi > lo, hi - lo, 1.0)
        anchor_space = (anchors - lo) / scale
        tree = cKDTree(anchor_space)
        if len(anchors) > 1:
            neighbor_distances, _ = tree.query(anchor_space, k=2)
            support_limit = float(np.quantile(neighbor_distances[:, 1], 0.95))
        else:
            support_limit = 0.0

        template = {**soil_params, **dict(zip(features, anchors[0]))}
        initial = self.consultant.preprocess(pd.DataFrame([template]), is_training=False)
        numerical = FEATURE_CONFIG['numerical']
        scale_idx = [numerical.index(feature) for feature in features]
        means = self.consultant.scaler.mean_[scale_idx]
        stds = self.consultant.scaler.scale_[scale_idx]

        def preference(candidates):
            candidates = np.atleast_2d(candidates)
            encoded = pd.DataFrame(
                np.repeat(initial.to_numpy(), len(candidates), axis=0), columns=initial.columns
            )
            encoded[features] = (candidates - means) / stds
            ie = self.consultant.model.predict(encoded)
            rate, temp, n_content, _ = candidates.T
            cost = 0.5 * (temp / 800 + rate / 10)
            fertility = n_content / 15
            return wie * ie / 100 - wco * cost + wfe * fertility

        def objective(values):
            distance, _ = tree.query((values - lo) / scale)
            if distance > support_limit + 1e-12:
                return 1e6 + float(distance - support_limit)
            return -float(preference(values)[0])

        scores = preference(anchors)
        best_observed = int(np.argmax(scores))
        result = differential_evolution(
            objective, bounds, strategy='best1bin', popsize=15, mutation=(0.5, 1),
            recombination=0.7, seed=42, x0=anchors[best_observed]
        )
        supported = tree.query((result.x - lo) / scale)[0] <= support_limit + 1e-12
        optimized = supported and np.isfinite(result.fun) and -result.fun > scores[best_observed]
        if not optimized:
            result.x = anchors[best_observed]
            result.fun = -float(scores[best_observed])
        result.success = bool(np.isfinite(result.fun))
        result.support_scope = scope
        result.support_count = len(anchors)
        result.used_observed_recipe = not optimized
        return result


# =================================================================

# =================================================================
class FeasibilityCalibrator:
    def __init__(self, consultant):
        self.consultant = consultant;
        self.calibration_df = None
        self.threshold_warning = None;
        self.calibration_error = None
        self.separable = None  # [必须1] 三级分离是否成立
        self.best_weights = None
        self.weight_search_results = None
        self.weight_calibration_metrics = None
        self.threshold_details = None

    @staticmethod
    def _distance_reference(train_array, top_k=5, n_ref=200):
        if len(train_array) < 2 or top_k < 1:
            raise ValueError("近邻评分至少需要两条训练记录和一个近邻。")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_array), size=min(n_ref, len(train_array)), replace=False)
        k = min(top_k, len(train_array) - 1)
        tree = cKDTree(train_array)
        distances, _ = tree.query(train_array[idx], k=k + 1)
        return tree, np.sort(distances[:, 1:].mean(axis=1)), k

    @staticmethod
    def _neighbor_scores(reference, values):
        tree, reference_distances, k = reference
        distances, indices = tree.query(np.atleast_2d(values), k=list(range(1, k + 1)))
        ranks = np.searchsorted(reference_distances, distances.mean(axis=1), side='right')
        scores = 100 * (1 - ranks / len(reference_distances))
        return scores, distances, indices

    @staticmethod
    def _boundary_quantiles(train_raw):
        return train_raw[ScenarioResponseAnalyzer.DESIGN_VARS].quantile([0.05, 0.95])

    def build_calibration_dataset(self, df_raw, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        target = FEATURE_CONFIG['target']
        features = ScenarioResponseAnalyzer.DESIGN_VARS
        samples = []
        prep = BiocharConsultant()
        for fold_number, (train_idx, val_idx) in enumerate(kf.split(df_raw), 1):
            train = df_raw.iloc[train_idx]
            val = df_raw.iloc[val_idx]
            try:
                X_train, y_train = prep.preprocess(train, is_training=True)
                X_val = prep.preprocess(val, is_training=False)
            except ValueError as exc:
                raise ValueError(f"OOF 第 {fold_number} 折预处理失败：{exc}") from exc
            model = prep._create_model(self.consultant.model_name, self.consultant.best_params)
            model.fit(X_train, y_train)
            true = val[target].to_numpy()
            pred = model.predict(X_val)
            # Use the same fold-fitted imputed values as the live scoring path.
            raw_numeric = prep.scaler.inverse_transform(X_val[FEATURE_CONFIG['numerical']])
            val_design = pd.DataFrame(raw_numeric, columns=FEATURE_CONFIG['numerical'])[features]
            quantiles = self._boundary_quantiles(train)
            within = val_design.ge(quantiles.loc[0.05]) & val_design.le(quantiles.loc[0.95])
            reference = self._distance_reference(X_train.to_numpy())
            distance, _, neighbors = self._neighbor_scores(reference, X_val.to_numpy())
            deviation = stabilized_relative_error(y_train[neighbors].mean(axis=1), pred)
            samples.append(pd.DataFrame({
                'boundary_score': within.sum(axis=1).to_numpy() * 25,
                'deviation_score': np.clip(120 - 2 * deviation, 0, 100),
                'distance_score': distance,
                'true_ie': true, 'pred_ie': pred, 'abs_error': np.abs(true - pred),
                'rel_error': stabilized_relative_error(true, pred),
            }))
        self.calibration_df = pd.concat(samples, ignore_index=True)
        return self.calibration_df

    def optimize_weights(self, objective='spearman', step=0.02, verbose=False):
        """
        基于5-fold OOF相对预测误差自动校准可行性权重。

        Composite feasibility score:
            F_fea = w_B * S_B + w_N * S_N + w_D * S_D

        其中：
            S_B = boundary_score
            S_N = distance_score (ECDF neighborhood support)
            S_D = deviation_score

        目标：
            min rho_s(F_fea, e_OOF)

        即希望可行性得分越高，OOF相对误差总体越低。
        权重在非负单纯形上以确定性网格搜索：
            w_B >= 0, w_N >= 0, w_D >= 0,
            w_B + w_N + w_D = 1

        注意：最优组合对应的Spearman p值是在同一数据上选权重后得到的，
        因此只作为探索性描述，不作为独立显著性证据。
        """
        if self.calibration_df is None:
            raise ValueError("请先构建校准数据集")

        if step <= 0 or step > 1:
            raise ValueError("step必须位于(0, 1]范围内。")

        n_steps = int(round(1.0 / step))
        if not np.isclose(n_steps * step, 1.0, atol=1e-10):
            raise ValueError("step应能整除1，例如0.01、0.02、0.05。")

        required = ['boundary_score', 'distance_score', 'deviation_score', 'rel_error']
        missing = [c for c in required if c not in self.calibration_df.columns]
        if missing:
            raise ValueError(f"校准数据缺少必要字段: {missing}")

        df = self.calibration_df[required].copy()
        for c in required:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        if len(df) < 10:
            raise ValueError(f"有效校准样本仅{len(df)}条，不足以稳定校准权重。")

        S_B = df['boundary_score'].to_numpy(dtype=float)
        S_N = df['distance_score'].to_numpy(dtype=float)
        S_D = df['deviation_score'].to_numpy(dtype=float)
        e_oof = df['rel_error'].to_numpy(dtype=float)

        rows = []
        best = None

        # 用整数步数生成单纯形，避免np.arange浮点累计误差
        for i in range(n_steps + 1):
            w_B = i / n_steps
            for j in range(n_steps - i + 1):
                w_N = j / n_steps
                w_D = 1.0 - w_B - w_N
                if w_D < -1e-12:
                    continue
                w_D = max(0.0, w_D)

                F_fea = w_B * S_B + w_N * S_N + w_D * S_D

                # 若综合分数无方差，则Spearman无意义
                if np.nanstd(F_fea) < 1e-12:
                    continue

                rho, p_value = spearmanr(F_fea, e_oof)
                if not np.isfinite(rho):
                    continue

                row = {
                    'w_B': float(w_B),
                    'w_N': float(w_N),
                    'w_D': float(w_D),
                    'rho_s': float(rho),
                    'p_value': float(p_value) if np.isfinite(p_value) else np.nan
                }
                rows.append(row)

                if best is None or rho < best['rho_s']:
                    best = row.copy()

        if best is None or len(rows) == 0:
            raise ValueError("校准得分或误差缺少有效变异，无法确定可行性权重。")

        search_results = (
            pd.DataFrame(rows)
            .sort_values(['rho_s', 'w_B', 'w_N', 'w_D'], ascending=[True, True, True, True])
            .reset_index(drop=True)
        )
        search_results.insert(0, 'Rank', np.arange(1, len(search_results) + 1))

        best_weights = {
            'boundary': round(best['w_B'], 6),
            'distance': round(best['w_N'], 6),
            'deviation': round(best['w_D'], 6)
        }

        # 保存到calibrator
        self.best_weights = best_weights.copy()
        self.weight_search_results = search_results.copy()
        self.weight_calibration_metrics = {
            'rho_s': float(best['rho_s']),
            'p_value': float(best['p_value']) if np.isfinite(best['p_value']) else np.nan,
            'step': float(step),
            'n_valid': int(len(df)),
            'n_combinations': int(len(search_results))
        }

        if verbose:
            print('=' * 60)
            print('Feasibility-weight calibration')
            print('=' * 60)
            print(f"Valid calibration samples: {len(df)}")
            print(f"Grid step: {step:.3f}")
            print(f"Tested combinations: {len(search_results)}")
            print('-' * 60)
            print(f"Best w_B = {best['w_B']:.2f}")
            print(f"Best w_N = {best['w_N']:.2f}")
            print(f"Best w_D = {best['w_D']:.2f}")
            print(f"Best Spearman rho = {best['rho_s']:.4f}")
            print(f"Associated p-value = {best['p_value']:.4g}")
            print('=' * 60)
            print('Note: p-value is post-selection and exploratory only.')

        return best_weights

    def _get_calibrated_weights(self):
        """若尚未进行权重搜索，则按论文设定step=0.02自动完成。"""
        if getattr(self, 'best_weights', None) is None:
            return self.optimize_weights(step=0.02)
        return self.best_weights

    # [必须1+2] 阈值校准：最小样本量保护 + separable标志
    def calibrate_thresholds(self, error_high=8, error_medium=15):
        if self.calibration_df is None:
            raise ValueError("请先构建校准数据集")

        if not 0 <= error_high < error_medium:
            raise ValueError("可信组误差阈值必须低于风险组误差阈值。")
        df = self.calibration_df.copy()
        w = self._get_calibrated_weights()
        df['total_score'] = (
            w['boundary'] * df['boundary_score'] +
            w['distance'] * df['distance_score'] +
            w['deviation'] * df['deviation_score']
        )

        N = len(df)
        # 对应 n_min = max(5, ceil(0.05N))
        n_min = max(5, int(np.ceil(0.05 * N)))
        credible = df[df['rel_error'] <= error_high]
        risky = df[df['rel_error'] > error_medium]

        self.calibration_error = None
        self.threshold_warning = None
        self.separable = None
        self.threshold_details = {
            'N': int(N),
            'n_min': int(n_min),
            'credible_n': int(len(credible)),
            'risky_n': int(len(risky)),
            'error_high': float(error_high),
            'error_medium': float(error_medium)
        }

        if len(credible) < n_min or len(risky) < n_min:
            self.calibration_error = (
                f"校准样本不足：可信样本(误差≤{error_high}%) {len(credible)} 条，"
                f"风险样本(误差>{error_medium}%) {len(risky)} 条，均需≥{n_min}条"
                f"（max(5, ceil(5%×{N}))）。当前数据无法支持可靠的三级阈值校准。"
            )
            self.separable = False
            return {'high': None, 'medium': None, 'separable': False}

        high_thresh = float(credible['total_score'].quantile(0.05))
        medium_thresh = float(risky['total_score'].quantile(0.95))
        self.threshold_details.update({
            'T_H_raw': high_thresh,
            'T_M_raw': medium_thresh
        })

        if high_thresh <= medium_thresh:
            self.threshold_warning = (
                f"校准数据未形成稳定的高、中、低可靠性分离："
                f"T_H={high_thresh:.1f} ≤ T_M={medium_thresh:.1f}。"
                f"当前评分体系对可信/风险样本区分度不足，因此不进行三级可靠性赋值；"
                f"建议报告连续F_fea和邻域支持信息。"
            )
            self.separable = False
            return {
                'high': high_thresh,
                'medium': medium_thresh,
                'separable': False
            }

        self.separable = True
        return {
            'high': high_thresh,
            'medium': medium_thresh,
            'separable': True
        }

    def evaluate_calibration(self, error_high=8, error_medium=15):
        if self.calibration_df is None:
            raise ValueError("请先构建校准数据集")

        df = self.calibration_df.copy()
        w = self._get_calibrated_weights()
        thresholds = self.calibrate_thresholds(
            error_high=error_high,
            error_medium=error_medium
        )

        df['total_score'] = (
            w['boundary'] * df['boundary_score'] +
            w['distance'] * df['distance_score'] +
            w['deviation'] * df['deviation_score']
        )

        rho, p_value = spearmanr(df['total_score'], df['rel_error'])

        # 优先采用网格搜索时保存的最优rho/p，二者理论上应与这里一致
        search_metrics = getattr(self, 'weight_calibration_metrics', {}) or {}
        best_rho = search_metrics.get('rho_s', rho)
        best_p = search_metrics.get('p_value', p_value)

        metrics = {
            'spearman_correlation': round(float(best_rho), 4),
            'spearman_p_value': float(best_p) if np.isfinite(best_p) else np.nan,
            'separable': thresholds.get('separable', False),
            'weight_grid_step': search_metrics.get('step', 0.02),
            'weight_combinations_tested': search_metrics.get('n_combinations', None),
            'calibration_n': search_metrics.get('n_valid', len(df))
        }

        if thresholds.get('separable', False):
            hm = df['total_score'] >= thresholds['high']
            lm = df['total_score'] < thresholds['medium']
            mm = ~hm & ~lm
            metrics.update({
                'high_level_mape': round(df.loc[hm, 'rel_error'].mean(), 2),
                'mid_level_mape': round(df.loc[mm, 'rel_error'].mean(), 2),
                'low_level_mape': round(df.loc[lm, 'rel_error'].mean(), 2),
                'high_level_count': int(hm.sum()),
                'mid_level_count': int(mm.sum()),
                'low_level_count': int(lm.sum())
            })
        else:
            metrics.update({
                'high_level_mape': None,
                'mid_level_mape': None,
                'low_level_mape': None,
                'high_level_count': 0,
                'mid_level_count': 0,
                'low_level_count': 0
            })

        return metrics



# =================================================================
#
# =================================================================
class BiocharConsultant:
    def __init__(self):
        self.model = None
        self.model_name = ""
        self.best_params = None
        self.onehot = None
        self.cat_feature_names = []
        self.cat_modes = {}
        self.scaler = StandardScaler()
        self.imputer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        self.is_trained = False
        self.raw_train_df = None
        self.scaled_train_X = None
        self.shap_values = None
        self.learning_curve_data = None
        self.scenario_figure_exports = {}
        self.distance_reference = None
        self._reset_calibration()
        self.automl_results = None
        self.studywise_results = None
        self.studywise_metrics = None
        self.studywise_split_summary = None

    def _reset_calibration(self):
        self.calibrator = None
        self.calibration_output = None
        self.feasibility_weights = {'boundary': 0.4, 'deviation': 0.4, 'distance': 0.2}
        self.feasibility_thresholds = {'high': None, 'medium': None}
        self.feasibility_separable = False
        self.feasibility_calibration = None
        self.feasibility_weight_search = None

    def validate_prediction_inputs(self, frame):
        clean, logs = DataGuardian().validate_and_clean(frame, training=False)
        for col, categories in zip(FEATURE_CONFIG['categorical'], self.onehot.categories_):
            unseen = clean[col].notna() & ~clean[col].isin(categories)
            if unseen.any():
                values = ', '.join(clean.loc[unseen, col].unique()[:8])
                rows = '、'.join(map(str, np.flatnonzero(unseen.to_numpy())[:8] + 1))
                raise ValueError(f"{col} 含训练时未见类别（数据行 {rows}）：{values}")
        return clean, logs

    @staticmethod
    def _create_model(model_type, params):
        if model_type == "RandomForest":
            return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        if model_type == "XGBoost":
            return XGBRegressor(**params, random_state=42, n_jobs=-1)
        if model_type == "LightGBM":
            return LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=-1)
        if model_type == "SVR":
            return SVR(**params)
        raise ValueError(f"未知模型：{model_type}")

    def log_message(self, message, level="info"):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        if level == "info":
            st.write(f"📝 [{now}] INFO: {message}")
        elif level == "success":
            st.success(f"✅ [{now}] SUCCESS: {message}")
        elif level == "error":
            st.error(f"❌ [{now}] ERROR: {message}")

    def generate_synthetic_data(self, n_samples=500):
        """⚠️ 仅用于系统功能测试，不可用于论文结论。"""
        np.random.seed(42)
        data = {
            'Soil_pH': np.random.uniform(4.0, 9.0, n_samples), 'Soil_Clay': np.random.uniform(5, 60, n_samples),
            'Soil_SOM': np.random.uniform(5, 80, n_samples), 'Soil_CEC': np.random.uniform(2, 50, n_samples),
            'Soil_Moisture': np.random.uniform(10, 40, n_samples), 'BC_Rate': np.random.uniform(0.5, 10.0, n_samples),
            'BC_Temp': np.random.uniform(300, 800, n_samples), 'BC_SSA': np.random.uniform(20, 800, n_samples),
            'BC_N': np.random.uniform(0.1, 15.0, n_samples), 'BC_Ash': np.random.uniform(5, 50, n_samples),
            'Metal_Total': np.random.uniform(20, 1500, n_samples), 'Metal_Available': None,
            'BC_Feedstock': np.random.choice(['Straw', 'Wood', 'Manure', 'Sludge'], n_samples),
            'Metal_Type': np.random.choice(['Cd', 'Pb', 'Cu', 'Zn', 'As'], n_samples),
            'Soil_Type': np.random.choice(['Red_Soil', 'Paddy_Soil', 'Black_Soil'], n_samples)}
        data['Metal_Available'] = data['Metal_Total'] * np.random.uniform(0.1, 0.6, n_samples)
        df = pd.DataFrame(data)
        ie = (df['BC_N'] * 1.5 + df['BC_Rate'] * 3.5 + (9 - df['Soil_pH']) * 2 +
              df['BC_Temp'] * 0.02 + np.random.normal(0, 5, n_samples))
        df[FEATURE_CONFIG['target']] = np.clip(ie, 5, 98);
        return df

    def preprocess(self, df, is_training=True):
        df_proc = df.copy()
        num_cols = FEATURE_CONFIG['numerical']
        cat_cols = FEATURE_CONFIG['categorical']
        missing = [col for col in num_cols + cat_cols if col not in df_proc.columns]
        if missing:
            raise ValueError("输入缺少字段：" + "、".join(missing))
        if is_training:
            empty = [col for col in num_cols if df_proc[col].isna().all()]
            if empty:
                raise ValueError("训练子集内数值列全缺失：" + "、".join(empty))
        if is_training:
            self.cat_modes = {}
            for col in cat_cols:
                ms = df_proc[col].mode()
                mv = ms.iloc[0] if not ms.empty else 'Unknown'
                self.cat_modes[col] = mv;
                df_proc[col] = df_proc[col].fillna(mv).astype(str)
        else:
            for col in cat_cols:
                df_proc[col] = df_proc[col].fillna(self.cat_modes.get(col, 'Unknown')).astype(str)
        if is_training:
            self.imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42, tol=1e-3)
            ni = self.imputer.fit_transform(df_proc[num_cols])
            self.scaler = StandardScaler();
            ns = self.scaler.fit_transform(ni)
        else:
            ni = self.imputer.transform(df_proc[num_cols]);
            ns = self.scaler.transform(ni)
        if is_training:
            self.onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ce = self.onehot.fit_transform(df_proc[cat_cols])
            self.cat_feature_names = self.onehot.get_feature_names_out(cat_cols).tolist()
        else:
            ce = self.onehot.transform(df_proc[cat_cols])
        X = np.hstack([ns, ce])
        feature_names = num_cols + self.cat_feature_names
        if is_training:
            self.feature_names = feature_names
        X = pd.DataFrame(X, columns=feature_names, index=df_proc.index)
        if is_training and FEATURE_CONFIG['target'] in df_proc.columns:
            return X, df_proc[FEATURE_CONFIG['target']].values
        return X

    def _prepare_cv_folds(self, df_train_raw, n_splits=5, groups=None):
        df_train_raw = df_train_raw.reset_index(drop=True)
        if groups is None:
            split_iter = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(df_train_raw)
        else:
            groups = pd.Series(groups).reset_index(drop=True)
            if len(groups) != len(df_train_raw) or groups.isna().any():
                raise ValueError("Study_ID 缺失或与训练记录数不一致。")
            if groups.nunique() < n_splits:
                raise ValueError(f"独立 Study_ID 少于所需的 {n_splits} 折。")
            split_iter = GroupKFold(n_splits=n_splits).split(df_train_raw, groups=groups)

        folds = []
        prep = BiocharConsultant()
        target = FEATURE_CONFIG['target']
        for fold_number, (tr_idx, va_idx) in enumerate(split_iter, 1):
            if len(va_idx) < 2:
                raise ValueError(f"第 {fold_number} 折验证集少于两条记录，无法计算 R²。请调整折数或数据划分。")
            train = df_train_raw.iloc[tr_idx]
            val = df_train_raw.iloc[va_idx]
            try:
                X_train, y_train = prep.preprocess(train, is_training=True)
                X_val = prep.preprocess(val, is_training=False)
            except ValueError as exc:
                raise ValueError(f"第 {fold_number} 折预处理失败：{exc}") from exc
            folds.append((X_train, y_train, X_val, val[target].to_numpy()))
        return folds

    def _optuna_objective_factory(self, folds, model_type):
        def objective(trial):
            if model_type == "RandomForest":
                p = {'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                     'max_depth': trial.suggest_int('max_depth', 5, 40),
                     'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                     'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                     'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])}
            elif model_type == "XGBoost":
                p = {'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                     'max_depth': trial.suggest_int('max_depth', 3, 15),
                     'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                     'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                     'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                     'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                     'n_estimators': trial.suggest_int('n_estimators', 100, 500)}
            elif model_type == "LightGBM":
                p = {'num_leaves': trial.suggest_int('num_leaves', 20, 256),
                     'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                     'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                     'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                     'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                     'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                     'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                     'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                     'n_estimators': trial.suggest_int('n_estimators', 100, 500)}
            else:
                p = {'C': trial.suggest_float('C', 0.01, 1000.0, log=True),
                     'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                     'epsilon': trial.suggest_float('epsilon', 0.001, 1.0, log=True)}
            scores = []
            for X_train, y_train, X_val, y_val in folds:
                model = self._create_model(model_type, p)
                model.fit(X_train, y_train)
                scores.append(r2_score(y_val, model.predict(X_val)))
            return float(np.mean(scores))

        return objective

    def _finalize_model(self, train_raw, test_raw, model_type, best_params):
        """Fit locally so a failed training run cannot overwrite the active model."""
        if len(test_raw) < 2:
            raise ValueError("独立测试集少于两条记录，无法计算 R²。")
        prep = BiocharConsultant()
        X_train, y_train = prep.preprocess(train_raw, is_training=True)
        X_test = prep.preprocess(test_raw, is_training=False)
        y_test = test_raw[FEATURE_CONFIG['target']].to_numpy()
        model = self._create_model(model_type, best_params)
        model.fit(X_train, y_train)

        for key in ('onehot', 'scaler', 'imputer', 'cat_modes', 'feature_names', 'cat_feature_names'):
            setattr(self, key, getattr(prep, key))
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.raw_train_df = train_raw.copy()
        self.scaled_train_X = X_train.reset_index(drop=True)
        self.best_params = best_params.copy()
        self.model = model
        self.model_name = model_type
        self._reset_calibration()
        self.distance_reference = None
        self.shap_values = None
        self.learning_curve_data = None
        self.scenario_figure_exports.clear()
        self.is_trained = True

    def train_with_optuna(self, df_raw, model_type, n_trials=20):
        """单模型Optuna训练"""
        train_raw, test_raw = train_test_split(df_raw, test_size=0.2, random_state=42)
        train_raw = train_raw.reset_index(drop=True);
        test_raw = test_raw.reset_index(drop=True)
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        folds = self._prepare_cv_folds(train_raw)
        study.optimize(self._optuna_objective_factory(folds, model_type), n_trials=n_trials)
        self.log_message(f"最佳参数: {study.best_params}")
        self._finalize_model(train_raw, test_raw, model_type, study.best_params)
        self.automl_results = None

    # [必须4] 四模型Optuna自动比较与CV择优
    def run_automl_selection(self, df_raw, n_trials=20):
        """
        真正的AutoML模型选择：
        1. 先80:20划分原始数据（测试集完全隔离）
        2. 对4个候选模型分别执行Optuna超参数优化（fold-safe 5折CV）
        3. 按平均CV R²选择最优模型 M*
        4. 用M*的最优超参数在全部训练集上训练最终模型
        5. 独立测试集仅评价一次
        """
        train_raw, test_raw = train_test_split(df_raw, test_size=0.2, random_state=42)
        train_raw = train_raw.reset_index(drop=True)
        test_raw = test_raw.reset_index(drop=True)
        folds = self._prepare_cv_folds(train_raw)
        results = []
        for i, mt in enumerate(AUTOML_MODELS):
            st.write(f"🔄 [{i + 1}/4] 正在优化 {mt} ...")
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(self._optuna_objective_factory(folds, mt), n_trials=n_trials)
            cv_r2 = float(study.best_value)
            results.append({
                'Model': mt,
                'CV_R²_raw': cv_r2,
                'CV_R²': round(cv_r2, 4),
                'Best_Params': study.best_params
            })
            st.write(f"   ✅ {mt} CV R² = {cv_r2:.4f}")
        result_df = pd.DataFrame(results)
        best_row = result_df.loc[result_df['CV_R²_raw'].idxmax()]
        best_model = best_row['Model']
        best_params = best_row['Best_Params']
        self.log_message(f"AutoML选择最优模型: {best_model} (CV R²={best_row['CV_R²_raw']:.6f})")
        self._finalize_model(train_raw, test_raw, best_model, best_params)
        self.automl_results = result_df
        return self.automl_results, best_model

    def run_studywise_sensitivity(self, df_raw, n_trials=20, group_col="Study_ID",
                                  test_size=0.20, n_splits=5):
        """
        Study-wise sensitivity analysis.

        Purpose
        -------
        Evaluate whether model performance remains stable when all rows from
        the same literature source are kept together. This analysis is
        separate from the primary record-wise model and DOES NOT overwrite
        the model used for SHAP, inverse design, or feasibility calibration.

        Procedure
        ---------
        1) Validate Study_ID.
        2) GroupShuffleSplit creates an independent study-level test subset.
        3) Within the grouped training subset, each candidate algorithm is
           tuned with Optuna + GroupKFold.
        4) The best algorithm is refitted on the full grouped training subset.
        5) The held-out study groups are evaluated once.
        6) The original primary-model state is restored automatically.
        """
        if group_col not in df_raw.columns:
            raise ValueError(
                f"数据中缺少'{group_col}'列。请先为每条记录建立可追溯的Study_ID。"
            )

        work = df_raw.copy().reset_index(drop=True)

        # Normalize group labels without turning genuine missing values into "nan".
        raw_groups = work[group_col]
        missing_mask = raw_groups.isna() | raw_groups.astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
        if missing_mask.any():
            examples = work.loc[missing_mask].index[:10].tolist()
            raise ValueError(
                f"{group_col}有{int(missing_mask.sum())}条缺失/空白记录，"
                f"示例行索引={examples}。来源未补齐前不要运行study-wise分析。"
            )

        work[group_col] = raw_groups.astype(str).str.strip()
        n_total_groups = int(work[group_col].nunique())
        if n_total_groups < (n_splits + 1):
            raise ValueError(
                f"当前只有{n_total_groups}个独立Study_ID。"
                f"至少需要{n_splits + 1}个，才能同时保留独立分组测试集并执行"
                f"{n_splits}折GroupKFold。"
            )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=42
        )
        train_idx, test_idx = next(
            splitter.split(work, groups=work[group_col])
        )

        train_raw = work.iloc[train_idx].reset_index(drop=True)
        test_raw = work.iloc[test_idx].reset_index(drop=True)
        train_groups = train_raw[group_col].astype(str).reset_index(drop=True)
        test_groups = test_raw[group_col].astype(str).reset_index(drop=True)

        overlap = set(train_groups.unique()) & set(test_groups.unique())
        if overlap:
            raise RuntimeError(
                "Study-wise划分失败：训练集与测试集存在重复Study_ID："
                + ", ".join(sorted(overlap))
            )

        if train_groups.nunique() < n_splits:
            raise ValueError(
                f"GroupShuffleSplit后训练集仅剩{train_groups.nunique()}个Study_ID，"
                f"无法进行{n_splits}折GroupKFold。请减少折数或调整测试比例。"
            )

        folds = self._prepare_cv_folds(train_raw, n_splits=n_splits, groups=train_groups)
        # Store a human-readable split summary.
        split_summary = {
            "Total_records": int(len(work)),
            "Total_studies": n_total_groups,
            "Train_records": int(len(train_raw)),
            "Test_records": int(len(test_raw)),
            "Train_studies": int(train_groups.nunique()),
            "Test_studies": int(test_groups.nunique()),
            "Group_overlap": 0,
            "CV_folds": n_splits
        }

        analysis = BiocharConsultant()
        results = []
        for i, model_type in enumerate(AUTOML_MODELS):
            st.write(f"🔬 Study-wise [{i + 1}/{len(AUTOML_MODELS)}] 正在优化 {model_type} ...")
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(analysis._optuna_objective_factory(folds, model_type), n_trials=n_trials)
            score = float(study.best_value)
            results.append({
                "Model": model_type, "Studywise_CV_R²_raw": score,
                "Studywise_CV_R²": round(score, 4), "Best_Params": study.best_params,
            })
            st.write(f"✅ {model_type} study-wise CV R² = {score:.4f}")

        result_df = pd.DataFrame(results)
        best = result_df.loc[result_df["Studywise_CV_R²_raw"].idxmax()]
        analysis._finalize_model(train_raw, test_raw, best["Model"], best["Best_Params"])
        y_pred = analysis.model.predict(analysis.X_test)
        y_true = analysis.y_test
        self.studywise_results = result_df
        self.studywise_metrics = {
            **split_summary,
            "Best_Model": best["Model"],
            "Best_CV_R²": float(best["Studywise_CV_R²_raw"]),
            "Test_R²": float(stable_r2(y_true, y_pred)),
            "Test_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "Test_MAE": float(mean_absolute_error(y_true, y_pred)),
        }
        self.studywise_split_summary = split_summary

        return self.studywise_results, self.studywise_metrics

    def run_feasibility_calibration(self, n_splits=5, error_high=8, error_medium=15, weight_step=0.02):
        """
        训练数据内部可行性校准：
        1) 5-fold OOF生成校准样本；
        2) 单纯形网格自动搜索w_B, w_N, w_D；
        3) 以min Spearman rho(F_fea, e_OOF)确定最终权重；
        4) 基于可信/风险OOF样本分布校准T_H/T_M。
        """
        if not self.is_trained:
            raise ValueError("请先训练模型")

        if not 2 <= n_splits <= len(self.raw_train_df):
            raise ValueError("校准折数超出训练记录数。")
        if not 0 <= error_high < error_medium:
            raise ValueError("可信组误差阈值必须低于风险组误差阈值。")
        calibrator = FeasibilityCalibrator(self)
        calibrator.build_calibration_dataset(self.raw_train_df, n_splits=n_splits)
        best_weights = calibrator.optimize_weights(step=weight_step)
        metrics = calibrator.evaluate_calibration(error_high=error_high, error_medium=error_medium)
        details = calibrator.threshold_details
        best_thresholds = {
            'high': details.get('T_H_raw'), 'medium': details.get('T_M_raw'),
            'separable': calibrator.separable,
        }

        self.calibrator = calibrator
        self.feasibility_weights = best_weights.copy()
        self.feasibility_separable = calibrator.separable
        self.feasibility_thresholds = {k: best_thresholds[k] for k in ('high', 'medium')}

        self.feasibility_calibration = {
            'w_B': best_weights['boundary'],
            'w_N': best_weights['distance'],
            'w_D': best_weights['deviation'],
            'rho_s': metrics['spearman_correlation'],
            'p_value': metrics.get('spearman_p_value', np.nan),
            'grid_step': weight_step,
            'n_combinations': metrics.get('weight_combinations_tested', None),
            'n_calibration': metrics.get('calibration_n', None),
            'separable': self.feasibility_separable,
            'T_H': best_thresholds.get('high', None),
            'T_M': best_thresholds.get('medium', None)
        }
        self.feasibility_weight_search = (
            self.calibrator.weight_search_results.copy()
            if self.calibrator.weight_search_results is not None
            else None
        )

        metrics.update(cv_folds=n_splits, error_high=error_high, error_medium=error_medium)
        return best_weights, best_thresholds, metrics

    # [建议8] 专家规则弱化过强确定性表述
    def generate_process_advice(self, best_rate, best_temp, best_n, best_ssa,
                                feedstock, metal_type, soil_ph, feasibility_result):
        """rule-based expert knowledge mapping"""
        advice = {}
        if best_temp < 400:
            advice['pyrolysis'] = (
                f"**低温热解（约{best_temp:.0f}℃）**：可考虑较慢升温速率（5~10℃/min）与较长保温时间（2~4小时），"
                "有助于保留较多表面含氧官能团（-COOH、-OH），可能更适合以络合吸附为主的修复场景。"
                "该温度区间产率通常较高，但生物炭芳香化程度相对较低，长期稳定性可能较弱，具体需结合原料特性验证。")
        elif 400 <= best_temp <= 600:
            advice['pyrolysis'] = (f"**中温热解（约{best_temp:.0f}℃）**：推荐升温速率10~15℃/min、保温1~2小时、限氧热解。"
                                   "该温度区间可能在孔隙发育与官能团保留之间取得平衡，兼具物理吸附与化学固定潜力。"
                                   "可考虑配备尾气冷凝装置回收木醋液等副产物，具体工艺参数需根据原料特性调整。")
        else:
            advice['pyrolysis'] = (
                f"**高温热解（约{best_temp:.0f}℃）**：建议采用程序升温，高温段保温60~90分钟以确保炭化充分。"
                "高温下生物炭比表面积与孔隙结构通常较发达，芳香化程度较高，长期稳定性可能较强。"
                "需配套尾气处理系统；同时应注意灰分熔融可能导致的孔隙堵塞问题，具体需结合原料灰分特性验证。")
        if best_rate < 2.0:
            advice['application'] = (f"**低施用量（{best_rate:.2f}%）**：不建议全田撒施，可考虑穴施或条施方式，"
                                     "集中施加于作物根际区域以提高局部修复浓度。施用深度可控制在15~20cm耕作层，"
                                     "适用于轻度污染农田的原位钝化，具体效果需通过小区实验验证。")
        elif 2.0 <= best_rate <= 5.0:
            advice['application'] = (f"**常规施用量（{best_rate:.2f}%）**：可考虑全田撒施后旋耕翻埋，翻耕深度20~30cm，"
                                     "确保生物炭与耕作层土壤均匀混合。可分两次施用（首次约70%，平衡1周后补施剩余部分），"
                                     "施用后保持田间含水率约60%，养护7~15天后再种植，具体养护周期需结合土壤条件调整。")
        else:
            advice['application'] = (
                f"**高施用量（{best_rate:.2f}%）**：建议分批次梯度施用，避免一次性大量施加导致土壤理化性质剧变。"
                "可考虑首年施加约60%，次年追施剩余部分；配合有机肥联合施用以缓冲土壤pH波动。"
                "高剂量场景下需重点监测土壤盐基离子变化，防止灰分过高导致的土壤板结风险。")
        metal_advice_map = {
            'Cd': "针对Cd污染：可考虑碱化改性（如钙基、镁基改性）以提升生物炭表面负电荷，增强静电吸附与沉淀作用；酸性土壤可配施石灰协同调理pH至6.5~7.0区间。",
            'Pb': "针对Pb污染：可考虑强化含氧官能团或负载磷酸盐、黏土矿物，通过络合与沉淀作用固定Pb；可配合磷源材料联合施用。",
            'Cu': "针对Cu污染：可考虑氨基化改性以提升含氮官能团含量，增强络合螯合能力；配合有机质腐殖化过程，促进Cu形成稳定有机结合态。",
            'Zn': "针对Zn污染：可考虑铁锰氧化物负载改性，利用氧化物表面吸附与共沉淀作用；注意控制土壤pH，避免过高pH下锌酸盐重新溶出的风险。",
            'As': "针对As污染：不建议单纯依赖高温生物炭，可考虑铁基改性（负载零价铁、水铁矿），将As(Ⅲ)氧化为As(Ⅴ)后通过专性吸附固定；需注意生物炭还原环境可能导致As活化的风险，配合好氧管理。"}
        advice['metal_target'] = metal_advice_map.get(metal_type,
                                                      "针对目标重金属，建议结合相似样本的实验条件开展验证性实验。")
        feedstock_advice_map = {
            'Straw': "秸秆基生物炭：原料灰分通常较高、钾素丰富，天然具备一定碱化效应。制备前建议粉碎至20~40目，可考虑水洗脱除部分可溶性盐，避免施用后土壤EC值过快上升。",
            'Wood': "木屑基生物炭：孔隙结构通常较发达、比表面积较高，灰分较低。制备时可适当延长保温时间以促进微孔发育；适合作为载体负载改性剂，提升靶向吸附能力。",
            'Manure': "畜禽粪便基生物炭：养分含量通常较高、磷钙丰富。热解温度不宜过低以确保病原菌降解；需注意控制原料重金属本底值，具体无害化温度应参照适用标准。",
            'Sludge': "污泥基生物炭：灰分与重金属本底通常较高，需严格控制热解条件以实现无害化。建议与秸秆、木屑按比例混合热解，以平衡孔隙与养分属性，具体配比需通过预实验确定。"}
        advice['feedstock'] = feedstock_advice_map.get(feedstock, "根据所选原料特性，合理调整热解工艺，确保炭化均匀。")
        level = feasibility_result['level']
        oob = [k for k, v in feasibility_result['boundary_check'].items() if not v['is_within']]
        if level.startswith("未"):
            advice['risk'] = (
                "**落地建议**：当前没有可用的三级分级。"
                "请结合连续得分、相似样本和候选配方来源安排预实验。"
            )
        elif level == "高可行性":
            advice['risk'] = "**落地建议**：该配方处于实验观测范围内，预测可信度较高。推荐直接开展小区验证实验，设置3个平行样，按推荐工艺施用后定期监测有效态含量变化。"
        elif level == "中等可行性":
            advice['risk'] = (f"**落地建议**：部分参数接近实验边界，存在一定不确定性。建议先开展实验室培养预实验，"
                              f"重点关注{'、'.join(oob) if oob else '边界参数'}的实际效果，根据预实验结果微调后再进行田间试验。")
        else:
            advice['risk'] = (f"**风险提示**：该配方存在明显模型外推风险，不建议直接投入实验。"
                              f"建议优先调整{'、'.join(oob) if oob else '核心参数'}至观测范围内，或补充对应条件的实验数据扩充训练集。")
        return advice

    def validate_formulation(self, formulation_dict, top_k=5):
        input_df, _ = self.validate_prediction_inputs(pd.DataFrame([formulation_dict]))
        input_scaled = self.preprocess(input_df, is_training=False)
        predicted_ie = self.model.predict(input_scaled)[0]
        raw_values = self.scaler.inverse_transform(input_scaled[FEATURE_CONFIG['numerical']])[0]
        raw_values = dict(zip(FEATURE_CONFIG['numerical'], raw_values))
        quantiles = FeasibilityCalibrator._boundary_quantiles(self.raw_train_df)
        boundary_results = {}
        for feat in ScenarioResponseAnalyzer.DESIGN_VARS:
            q5, q95 = quantiles.loc[0.05, feat], quantiles.loc[0.95, feat]
            val = raw_values[feat]
            within = q5 <= val <= q95
            boundary_results[feat] = {
                'value': round(val, 2), 'q5': round(q5, 2), 'q95': round(q95, 2), 'is_within': within,
            }
        boundary_score = 25 * sum(item['is_within'] for item in boundary_results.values())
        if self.distance_reference is None or self.distance_reference[0] != top_k:
            self.distance_reference = (
                top_k, FeasibilityCalibrator._distance_reference(self.scaled_train_X.to_numpy(), top_k)
            )
        scores, distances, neighbors = FeasibilityCalibrator._neighbor_scores(
            self.distance_reference[1], input_scaled.to_numpy()
        )
        distance_score = scores[0]
        similar_samples = self.raw_train_df.iloc[neighbors[0]].copy()
        similar_samples['特征距离'] = np.round(distances[0], 4)
        similar_ie_mean = similar_samples[FEATURE_CONFIG['target']].mean()
        similar_ie_std = similar_samples[FEATURE_CONFIG['target']].std()
        relative_deviation = float(stabilized_relative_error(similar_ie_mean, predicted_ie))
        deviation_score = float(np.clip(120 - 2 * relative_deviation, 0, 100))
        w = self.feasibility_weights
        total_score = boundary_score * w['boundary'] + deviation_score * w['deviation'] + distance_score * w['distance']
        # [必须1] 不可分离时不进行三级评级
        if self.calibrator is None:
            level = "未校准"
            risk_tip = "当前模型尚未执行 OOF 校准。连续得分基于默认权重，仅供参考。"
        elif not self.feasibility_separable:
            level = "未分级（校准分离不足）"
            risk_tip = "当前校准数据未形成稳定的高、中、低可靠性分离，不进行三级可靠性赋值。该得分仅供参考，建议结合相似实验样本人工判断。"
        else:
            ht, mt = self.feasibility_thresholds['high'], self.feasibility_thresholds['medium']
            if total_score >= ht:
                level, risk_tip = "高可行性", "配方处于实验观测范围内，达到当前校准的高支持阈值，可结合相似样本安排实验验证"
            elif total_score >= mt:
                level, risk_tip = "中等可行性", "部分参数接近实验边界，预测存在一定偏差，建议先开展小范围预实验"
            else:
                level, risk_tip = "低可行性", "存在明显模型外推风险，缺乏真实实验数据支撑，不建议直接采用该配方"
        return {'total_score': round(total_score, 1), 'level': level, 'risk_tip': risk_tip,
                'boundary_check': boundary_results, 'similar_samples': similar_samples,
                'similar_ie_mean': round(similar_ie_mean, 2), 'similar_ie_std': round(similar_ie_std, 2),
                'predicted_ie': predicted_ie, 'relative_deviation': round(relative_deviation, 2)}

    # [建议5] Baseline保存/恢复预处理状态
    def run_multi_model_benchmark(self, df_raw):
        # 保存正式模型的预处理状态快照
        saved = {k: getattr(self, k) for k in
                 ['onehot', 'scaler', 'imputer', 'cat_modes', 'feature_names', 'cat_feature_names',
                  'X_train', 'X_test', 'y_train', 'y_test', 'raw_train_df', 'scaled_train_X']}
        try:
            train_raw, test_raw = train_test_split(df_raw, test_size=0.2, random_state=42)
            train_raw = train_raw.reset_index(drop=True)
            test_raw = test_raw.reset_index(drop=True)
            X_train, y_train = self.preprocess(train_raw, is_training=True)
            X_test = self.preprocess(test_raw, is_training=False)
            y_test = test_raw[FEATURE_CONFIG['target']].values
            models = {'Linear Regression': LinearRegression(), 'SVR': SVR(kernel='rbf'),
                      'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                      'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                      'XGBoost': XGBRegressor(n_estimators=100, random_state=42)}
            ml, ts = [], []
            ref_std = np.std(y_test)
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = stable_r2(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)
                ml.extend([{'Model': name, 'Metric': 'R²', 'Value': r2},
                           {'Model': name, 'Metric': 'RMSE', 'Value': rmse},
                           {'Model': name, 'Metric': 'MAE', 'Value': mae}])
                try:
                    corr = np.corrcoef(y_test, preds)[0, 1]
                except:
                    corr = 0
                ts.append({'name': name, 'std': np.std(preds), 'corr': np.clip(corr, -1.0, 1.0)})
            return pd.DataFrame(ml), ts, ref_std
        finally:
            # [建议5] 恢复正式模型的预处理状态，避免baseline覆盖
            for k, v in saved.items(): setattr(self, k, v)


# =================================================================
#
# =================================================================
def activate_dataset(df, revision):
    if st.session_state.get('dataset_revision') != revision:
        st.session_state['consultant'] = BiocharConsultant()
        st.session_state['dataset_revision'] = revision
        st.session_state['current_df'] = df


def main():
    if 'guardian' not in st.session_state: st.session_state['guardian'] = DataGuardian()
    if 'history' not in st.session_state: st.session_state['history'] = HistoryManager()
    if 'consultant' not in st.session_state: st.session_state['consultant'] = BiocharConsultant()
    consultant = st.session_state['consultant']
    history_mgr = st.session_state['history']

    st.title("🌱 生物炭修复重金属污染土壤专家决策系统")
    st.caption(
        "融合 AutoML四模型择优、fold-safe预处理、SHAP/PDP解释、差分进化加权多目标优化与数据集实证校验 | 15项特征体系")

    with st.sidebar:
        st.header("🛠️ 功能面板")
        app_mode = st.radio("前往模块", ["系统介绍", "实验数据管理", "自动化建模分析",
                                         "单样本快速预测", "反向配方定制", "批量预测工具", "历史记录查询"])
        st.divider()
        st.info("系统状态: " + ("✅ 已加载模型" if consultant.is_trained else "⏳ 等待数据训练"))

    if app_mode == "系统介绍":
        st.markdown("""
                        ### 🔬 系统概况
                        本系统利用机器学习解决生物炭修复中的"场地异质性"问题，通过15个维度综合建模实现修复效果精准预测与材料按需定制。

                        #### 📈 核心技术流程
                        1. **数据驱动特征工程**：土壤pH, SOM, CEC, 含水率等12项连续特征 + 3项分类特征。Metal_Available 为施炭前初始有效态浓度，单位 mg/kg。
                        2. **AutoML四模型自动择优**：XGBoost、RandomForest、LightGBM、SVR分别经Optuna贝叶斯优化，自动选择最优模型。
                        3. **Fold-safe交叉验证**：每折仅用该折训练数据拟合迭代插补、标准化与One-Hot编码。
                        4. **独立测试集隔离**：先80:20划分原始数据，再在训练集上拟合全部预处理，测试集用于最终性能评价。
                        5. **SHAP + PDP可解释性**：Mean|SHAP|全局排名（分类变量grouped signed聚合）、SHAP dependence（原始物理单位）、PDP非线性响应分析。
                        6. **差分进化加权多目标优化**：在"固定效率-相对成本代理-肥力相关代理"之间搜索加权折衷最优配方。
                        7. **数据驱动可行性校准**：ECDF近邻距离+边界合规+预测偏差三维度，simplex grid search优化权重，仅用训练数据校准。
                        8. **三级可靠性分级**：仅当高可信阈值 > 风险阈值时进行三级评级；分离不足时如实报告，不强行分级。
                        9. **经验规则专家建议**：rule-based，根据推荐配方生成制备、施用、改性全流程工艺指导。

                        #### ⚠️ 数据使用说明
                        内置模拟数据集仅用于系统功能测试，目标函数为人为指定，SHAP分析结果不代表科学发现。论文结论必须基于真实文献数据库运行结果。
                        """)

    elif app_mode == "实验数据管理":
        st.header("📂 实验数据集管理")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("数据导入")
            uploaded_file = st.file_uploader("上传实验数据 (CSV)", type="csv")
            if uploaded_file:
                payload = uploaded_file.getvalue()
                revision = hashlib.sha256(payload).hexdigest()
                try:
                    df = pd.read_csv(
                        io.BytesIO(payload), dtype={'Study_ID': 'string'}, encoding='utf-8-sig'
                    )
                    clean_df, logs = st.session_state['guardian'].validate_and_clean(df)
                except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
                    st.error(f"导入失败：{exc}。当前数据集与模型仍保留。")
                else:
                    activate_dataset(clean_df, revision)
                    with st.expander("🛡️ 数据卫士：质量检测报告", expanded=True):
                        for log in logs:
                            st.info(log)
                        st.success("数据导入完成！")
            elif st.button("使用内置模拟数据集（仅功能测试）"):
                df = consultant.generate_synthetic_data(600)
                activate_dataset(df, "synthetic:600:42")
            if st.session_state.get('dataset_revision') == "synthetic:600:42":
                st.warning("⚠️ 当前为600组模拟样本，仅用于功能测试。")
        if 'current_df' in st.session_state:
            df = st.session_state['current_df']

            # ---------------------------------------------------------
            # Study-source audit (metadata only; never used as predictors)
            # ---------------------------------------------------------
            source_col = None
            if 'Source_Title' in df.columns:
                source_col = 'Source_Title'
            elif 'Unnamed: 17' in df.columns:
                source_col = 'Unnamed: 17'

            if source_col is not None:
                with st.expander("📚 文献来源 / Study_ID审计", expanded=False):
                    st.caption(
                        "该区域只检查研究来源元数据，不会把Source_Title或Study_ID作为模型预测变量。"
                    )
                    src_clean = (
                        df[source_col]
                        .astype(str)
                        .str.replace(r'\s+', ' ', regex=True)
                        .str.strip()
                    )
                    st.write(f"当前来源标签数量：**{src_clean.nunique()}**")
                    src_counts = (
                        src_clean.value_counts(dropna=False)
                        .rename_axis('Source_Title')
                        .reset_index(name='N_records')
                    )
                    st.dataframe(src_counts, use_container_width=True)

                    if 'Study_ID' in df.columns:
                        sid = df['Study_ID']
                        missing_sid = (
                            sid.isna()
                            | sid.astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
                        )
                        st.write(f"Study_ID数量：**{sid[~missing_sid].astype(str).nunique()}**")
                        st.write(f"Study_ID缺失/空白记录：**{int(missing_sid.sum())}**")
                        if int(missing_sid.sum()) == 0:
                            group_counts = (
                                df.groupby('Study_ID', dropna=False)
                                .size()
                                .sort_values(ascending=False)
                                .rename('N_records')
                                .reset_index()
                            )
                            st.dataframe(group_counts, use_container_width=True)
                            st.success("Study_ID已完整，可进入study-wise敏感性分析。")
                        else:
                            st.warning(
                                "Study_ID尚未补齐。请先追溯所有原始研究来源，"
                                "不要根据模糊来源标签自动推断Study_ID。"
                            )
                    else:
                        st.warning(
                            "当前数据中没有Study_ID列。请先依据可追溯的原始论文来源建立Study_ID，"
                            "再运行study-wise敏感性分析。"
                        )

            with col2:
                st.subheader("特征矩阵概览")

                paper_vars = FEATURE_CONFIG['numerical'] + [FEATURE_CONFIG['target']]

                paper_stats = df[paper_vars].describe().T

                # 增加IQR
                paper_stats['IQR'] = paper_stats['75%'] - paper_stats['25%']

                # 调整列顺序
                paper_stats = paper_stats[
                    ['count', 'mean', 'std', '50%', '25%', '75%', 'IQR', 'min', 'max']
                ]

                # 改成适合论文使用的列名
                paper_stats = paper_stats.rename(columns={
                    'count': 'N',
                    'mean': 'Mean',
                    'std': 'SD',
                    '50%': 'Median',
                    '25%': 'Q1',
                    '75%': 'Q3',
                    'min': 'Min',
                    'max': 'Max'
                })

                st.dataframe(
                    paper_stats.round(3),
                    use_container_width=True
                )

            st.divider()
            st.subheader("数据分布探索 (EDA)")
            vc1, vc2 = st.columns(2)
            with vc1:
                ftp = st.selectbox("选择特征", FEATURE_CONFIG['numerical'])
                fig, ax = plt.subplots()
                sns.histplot(df[ftp], kde=True, ax=ax, color=COLOR_BLUE, edgecolor='white', alpha=0.85,
                             line_kws={'color': COLOR_ORANGE, 'linewidth': 2})
                style_axis(ax)
                fig.tight_layout()
                st.pyplot(fig)
            with vc2:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[FEATURE_CONFIG['numerical']].corr(), cmap=DIVERGING_CMAP, center=0, vmin=-1, vmax=1,
                            ax=ax, linewidths=0.35, linecolor='white', cbar_kws={'label': 'Pearson correlation'})
                fig.tight_layout()
                st.pyplot(fig)

    elif app_mode == "自动化建模分析":
        st.header("🚀 自动化建模引擎 (AutoML)")
        if 'current_df' not in st.session_state:
            st.warning("请先在数据管理模块加载数据。")
        else:
            df = st.session_state['current_df']
            with st.expander("📊 多算法基准对比", expanded=False):
                st.write("注意：以下为默认超参数baseline，非Optuna优化后模型。运行后自动恢复正式模型预处理状态。")
                if st.button("运行基准对比"):
                    with st.spinner("正在训练baseline模型..."):
                        mdf, tstats, rstd = consultant.run_multi_model_benchmark(df)
                        viz = BiocharVisualizer()
                        c1, c2 = st.columns([1, 1])
                        with c1: st.write("#### 模型对比图（Baseline）"); st.pyplot(viz.plot_model_comparison_bar(mdf))
                        with c2: st.write("#### 泰勒图（Baseline）"); st.pyplot(viz.plot_taylor_diagram(tstats, rstd))
            st.divider()

            # [必须4] AutoML四模型自动比较入口
            st.subheader("🤖 AutoML 四模型自动择优（推荐）")
            st.caption(
                "依次对XGBoost、RandomForest、LightGBM、SVR执行Optuna超参数优化，按5折CV R²自动选择最优模型。独立测试集仅评价一次。")
            ac1, ac2 = st.columns(2)
            with ac1:
                automl_trials = st.slider("每个模型的Optuna迭代次数", 10, 100, 30, 5, key="automl_trials")
            with ac2:
                st.write("")
            if st.button("🚀 启动AutoML四模型自动择优"):
                with st.spinner("AutoML正在依次优化4个模型（fold-safe 5折CV）..."):
                    start_time = time.time()
                    try:
                        results_df, best_model = consultant.run_automl_selection(df, n_trials=automl_trials)
                        duration = time.time() - start_time
                        st.success(f"✅ AutoML完成！用时 {duration:.1f}秒，最优模型: {best_model}")
                        st.markdown("#### 📊 四模型CV性能对比")
                        st.dataframe(results_df[['Model', 'CV_R²']], use_container_width=True)
                    except ValueError as exc:
                        st.error(f"AutoML未完成：{exc}")
            st.divider()

            # =========================================================
            # Study-wise sensitivity analysis
            # =========================================================
            with st.expander("🧪 Study-wise敏感性分析（按原始研究来源分组）", expanded=False):
                st.caption(
                    "用途：检查同一篇论文贡献多条记录时，普通随机划分是否可能高估跨研究泛化性能。"
                    "Study_ID仅作为分组变量，不进入模型特征。该分析不会覆盖上方record-wise主模型。"
                )

                if 'Study_ID' not in df.columns:
                    st.warning(
                        "当前数据缺少Study_ID列。请先在原始数据中补充可追溯的Study_ID后再运行。"
                    )
                else:
                    sid = df['Study_ID']
                    sid_missing = (
                        sid.isna()
                        | sid.astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
                    )
                    n_studies = sid[~sid_missing].astype(str).str.strip().nunique()

                    swc1, swc2, swc3 = st.columns(3)
                    swc1.metric("有效Study_ID数量", int(n_studies))
                    swc2.metric("Study_ID缺失记录", int(sid_missing.sum()))
                    swc3.metric("总记录数", int(len(df)))

                    if int(sid_missing.sum()) > 0:
                        st.error(
                            "Study_ID未补齐，study-wise分析已锁定。"
                            "请先追溯缺失来源；程序不会自动猜测Study_ID。"
                        )
                    else:
                        group_counts = (
                            df.groupby(sid.astype(str).str.strip().rename('Study_ID'))
                            .size()
                            .sort_values(ascending=False)
                            .rename('N_records')
                            .reset_index()
                        )
                        st.write("##### 各研究来源贡献记录数")
                        st.dataframe(group_counts, use_container_width=True)

                        sw1, sw2, sw3 = st.columns(3)
                        with sw1:
                            study_trials = st.slider(
                                "每模型Optuna次数（study-wise）",
                                10, 100, 20, 5,
                                key="studywise_trials"
                            )
                        with sw2:
                            study_test_size = st.slider(
                                "按研究来源留出比例",
                                0.10, 0.40, 0.20, 0.05,
                                key="studywise_test_size"
                            )
                        with sw3:
                            study_folds = st.slider(
                                "GroupKFold折数",
                                3, 10, 5, 1,
                                key="studywise_folds"
                            )

                        if st.button("🔬 运行Study-wise敏感性分析", key="run_studywise"):
                            try:
                                with st.spinner(
                                    "正在按Study_ID分组：GroupShuffleSplit独立留出 + "
                                    "GroupKFold内部Optuna..."
                                ):
                                    start_time = time.time()
                                    sw_results, sw_metrics = consultant.run_studywise_sensitivity(
                                        df,
                                        n_trials=study_trials,
                                        group_col='Study_ID',
                                        test_size=study_test_size,
                                        n_splits=study_folds
                                    )
                                    duration = time.time() - start_time

                                st.success(
                                    f"✅ Study-wise敏感性分析完成，用时 {duration:.1f} 秒。"
                                    "主模型继续使用原有训练结果。"
                                )

                                st.write("##### 四模型Study-wise CV结果")
                                st.dataframe(
                                    sw_results[['Model', 'Studywise_CV_R²']],
                                    use_container_width=True
                                )

                                st.write("##### Study-wise独立测试结果")
                                sm1, sm2, sm3, sm4 = st.columns(4)
                                sm1.metric("最优模型", str(sw_metrics['Best_Model']))
                                sm2.metric("Test R²", f"{sw_metrics['Test_R²']:.4f}")
                                sm3.metric("Test RMSE", f"{sw_metrics['Test_RMSE']:.2f} 个百分点")
                                sm4.metric("Test MAE", f"{sw_metrics['Test_MAE']:.2f} 个百分点")

                                st.write("##### 分组划分信息")
                                split_df = pd.DataFrame([{
                                    'Total records': sw_metrics['Total_records'],
                                    'Total studies': sw_metrics['Total_studies'],
                                    'Train records': sw_metrics['Train_records'],
                                    'Test records': sw_metrics['Test_records'],
                                    'Train studies': sw_metrics['Train_studies'],
                                    'Test studies': sw_metrics['Test_studies'],
                                    'Study overlap': sw_metrics['Group_overlap'],
                                    'CV folds': sw_metrics['CV_folds']
                                }])
                                st.dataframe(split_df, use_container_width=True)

                                # Download-ready result table
                                export_sw = sw_results.copy()
                                export_sw['Best_Params'] = export_sw['Best_Params'].astype(str)
                                for k, v in sw_metrics.items():
                                    export_sw[f"Summary_{k}"] = v
                                st.download_button(
                                    "⬇️ 下载Study-wise敏感性分析结果CSV",
                                    data=export_sw.to_csv(index=False).encode('utf-8-sig'),
                                    file_name="Studywise_sensitivity_results.csv",
                                    mime="text/csv",
                                    key="download_studywise_results"
                                )

                                st.info(
                                    "注意：一次GroupShuffleSplit的结果取决于被留出的研究来源。"
                                    "该结果用于敏感性比较，不应自动替换原record-wise主分析。"
                                )

                            except Exception as e:
                                st.error(f"Study-wise分析未执行：{e}")

            st.divider()

            # 单模型手动选择（兼容原有方式）
            st.subheader("⚙️ 单模型手动优化（备选）")
            st.caption("如需手动指定模型进行Optuna优化，使用此入口。建议优先使用上方AutoML自动择优。")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                model_type = st.selectbox("算法选择", AUTOML_MODELS, key="manual_model")
            with mc2:
                n_trials = st.slider("贝叶斯优化迭代次数", 10, 200, 50, 5, key="manual_trials")
            with mc3:
                st.write("")
            if st.button("开始训练（单模型）"):
                with st.spinner("AI正在训练（先80:20划分，再fit预处理；fold-safe 5折CV）..."):
                    start_time = time.time()
                    try:
                        consultant.train_with_optuna(df, model_type, n_trials)
                        st.success(f"模型训练完成！用时: {time.time() - start_time:.2f}秒")
                    except ValueError as exc:
                        st.error(f"训练未完成：{exc}")
            if consultant.is_trained:
                st.divider();
                st.subheader("📉 模型性能评估（独立测试集）")
                if consultant.automl_results is not None:
                    st.info(f"当前模型由AutoML自动选择: **{consultant.model_name}**")
                y_pred = consultant.model.predict(consultant.X_test)
                pm1, pm2, pm3, pm4 = st.columns(4)
                test_r2 = stable_r2(consultant.y_test, y_pred)
                pm1.metric("决定系数 R²", f"{test_r2:.4f}" if np.isfinite(test_r2) else "未定义")
                pm2.metric("平均绝对误差 MAE", f"{mean_absolute_error(consultant.y_test, y_pred):.2f} 个百分点")
                pm3.metric("均方根误差 RMSE", f"{np.sqrt(mean_squared_error(consultant.y_test, y_pred)):.2f} 个百分点")
                mean_rel = stabilized_relative_error(consultant.y_test, y_pred).mean()
                pm4.metric("稳定化平均相对误差", f"{mean_rel:.2f}%",
                           help="分母取真实值绝对值与 1 个百分点中的较大值。")
                viz = BiocharVisualizer()
                regression_fig = viz.plot_regression_metrics(consultant.y_test, y_pred)
                st.pyplot(regression_fig)
                plt.close(regression_fig)

                st.divider();
                st.subheader("🔍 特征可解释性分析")
                st.caption(
                    "为保持独立测试集只用于最终泛化评价，论文中的全局SHAP排名、beeswarm与dependence plot均基于完整训练集计算。"
                )

                # ============================================================
                # Paper SHAP interpretation
                # - Final model is fitted on complete training set
                # - SHAP interpretation uses complete training set
                # - Independent test set is reserved for final performance only
                # ============================================================
                shap_v = None
                X_shap = None
                shap_agg = None

                if consultant.model_name in ["XGBoost", "RandomForest", "LightGBM"]:
                    X_shap = consultant.X_train
                    if consultant.shap_values is None:
                        explainer = shap.TreeExplainer(consultant.model)
                        consultant.shap_values = explainer.shap_values(X_shap)
                    shap_v = consultant.shap_values

                    # SHAP新版可能返回Explanation；统一转换为ndarray
                    if hasattr(shap_v, "values"):
                        shap_v = shap_v.values
                    shap_v = np.asarray(shap_v)

                    # 部分二分类/多输出实现可能多一维；回归任务原则上应为(n, p)
                    if shap_v.ndim != 2:
                        st.warning(
                            f"当前SHAP数组维度为 {shap_v.shape}，预期为二维(n_samples, n_features)。请检查解释器输出。")

                se1, se2 = st.columns(2)

                with se1:
                    st.write("#### Mean |SHAP| 排名（15个原始变量）")
                    if shap_v is not None and shap_v.ndim == 2:
                        cat_prefixes = ['BC_Feedstock_', 'Metal_Type_', 'Soil_Type_']
                        agg_rows, used = [], set()
                        fl = list(consultant.feature_names)

                        # 分类变量：先在每个样本内对One-Hot SHAP进行有符号聚合，再取平均绝对值
                        for prefix in cat_prefixes:
                            pname = prefix.rstrip('_')
                            cols = [i for i, f in enumerate(fl) if f.startswith(prefix)]
                            if cols:
                                grouped = np.abs(shap_v[:, cols].sum(axis=1)).mean()
                                agg_rows.append({'Feature': pname, 'Mean_|SHAP|': float(grouped)})
                                used.update(cols)

                        # 连续变量：mean absolute SHAP
                        for i, feat in enumerate(fl):
                            if i not in used:
                                agg_rows.append({
                                    'Feature': feat,
                                    'Mean_|SHAP|': float(np.abs(shap_v[:, i]).mean())
                                })

                        shap_agg = (
                            pd.DataFrame(agg_rows)
                            .sort_values('Mean_|SHAP|', ascending=False)
                            .reset_index(drop=True)
                        )
                        shap_agg.index = shap_agg.index + 1
                        shap_agg['Relative_Importance_%'] = (
                                100.0 * shap_agg['Mean_|SHAP|'] / shap_agg['Mean_|SHAP|'].sum()
                        )

                        # 显示时四舍五入，但后续计算始终保留原始值
                        display_shap_agg = shap_agg.copy()
                        display_shap_agg['Mean_|SHAP|'] = display_shap_agg['Mean_|SHAP|'].round(4)
                        display_shap_agg['Relative_Importance_%'] = display_shap_agg['Relative_Importance_%'].round(2)
                        st.dataframe(display_shap_agg, use_container_width=True)

                        # 前4项累计贡献，方便论文3.3直接引用
                        top4_share = shap_agg.head(4)['Relative_Importance_%'].sum()
                        st.info(f"前4项变量累计SHAP相对贡献：{top4_share:.2f}%")

                        fig_ms, ax_ms = plt.subplots(figsize=(8, 6.5))
                        shap_bar_colors = [feature_color(f) for f in shap_agg['Feature']]
                        ax_ms.barh(
                            np.arange(len(shap_agg)),
                            shap_agg['Mean_|SHAP|'].values,
                            color=shap_bar_colors,
                            edgecolor='white', linewidth=0.8, alpha=0.95
                        )
                        ax_ms.set_yticks(np.arange(len(shap_agg)))
                        ax_ms.set_yticklabels(shap_agg['Feature'].tolist())
                        ax_ms.invert_yaxis()
                        ax_ms.set_title("Mean |SHAP| Ranking", fontweight='bold', pad=10)
                        ax_ms.set_xlabel("Mean |SHAP| value")
                        ax_ms.set_ylabel("")
                        style_axis(ax_ms, grid_axis='x')
                        add_feature_family_legend(ax_ms, shap_agg['Feature'])
                        fig_ms.tight_layout()
                        st.pyplot(fig_ms)
                    else:
                        st.info("当前模型无法生成二维TreeSHAP结果。SVR需采用模型无关SHAP。")

                with se2:
                    st.write("#### SHAP Beeswarm Summary（仅连续变量）")
                    if shap_v is not None and shap_v.ndim == 2:
                        numerical_features = [
                            f for f in FEATURE_CONFIG['numerical']
                            if f in consultant.feature_names
                        ]
                        num_idx = [consultant.feature_names.index(f) for f in numerical_features]

                        # 从模型输入空间中提取连续变量，并逆标准化回原始物理单位
                        X_num = X_shap[numerical_features].copy()
                        shap_num = shap_v[:, num_idx].copy()
                        for feat in numerical_features:
                            model_idx = numerical_features.index(feat)
                            scaler_idx = FEATURE_CONFIG['numerical'].index(feat)
                            X_num.iloc[:, model_idx] = (
                                    X_num.iloc[:, model_idx] * consultant.scaler.scale_[scaler_idx]
                                    + consultant.scaler.mean_[scaler_idx]
                            )

                        plt.figure(figsize=(9, 7))
                        shap.summary_plot(
                            shap_num,
                            X_num,
                            feature_names=numerical_features,
                            show=False,
                            max_display=len(numerical_features),
                            cmap=SHAP_CMAP
                        )
                        plt.gca().set_facecolor("white")
                        plt.gca().tick_params(colors=COLOR_DARK)
                        plt.title("SHAP Summary of Continuous Predictors", fontweight='bold', pad=15)
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                        plt.close(plt.gcf())
                    else:
                        st.info("当前模型无法生成连续变量beeswarm。SVR需采用模型无关SHAP。")

                # 训练集原始缺失率检查：重点用于判断Soil_Moisture等高SHAP变量是否受插补结构影响
                with st.expander("🧪 SHAP稳健性检查：训练集预测变量缺失率", expanded=False):
                    raw_num = consultant.raw_train_df[FEATURE_CONFIG['numerical']].copy()
                    missing_summary = raw_num.isna().sum().to_frame("Missing_n")
                    missing_summary["Missing_%"] = 100.0 * missing_summary["Missing_n"] / len(raw_num)
                    missing_summary = missing_summary.sort_values("Missing_%", ascending=False)
                    missing_summary["Missing_%"] = missing_summary["Missing_%"].round(2)
                    st.dataframe(missing_summary, use_container_width=True)
                    st.caption(
                        "若高SHAP变量（尤其Soil_Moisture）缺失率较高，应在论文中谨慎解释，并建议进一步进行原始/插补样本敏感性分析。"
                    )

                # 内置feature_importances_（Supplementary，不作为论文主要指标）
                if consultant.model_name in ["XGBoost", "RandomForest", "LightGBM"]:
                    with st.expander("📎 树模型内置 feature_importances", expanded=False):
                        fig_fi = viz.plot_feature_importance(
                            consultant.model,
                            consultant.feature_names,
                            aggregate_onehot=True
                        )
                        if fig_fi:
                            st.pyplot(fig_fi)

                # SHAP dependence：论文解释同样基于完整训练集
                if shap_v is not None and shap_v.ndim == 2:
                    st.divider();
                    st.subheader("📊 SHAP Dependence Plot（训练集，原始物理单位）")
                    nf = [f for f in FEATURE_CONFIG['numerical'] if f in consultant.feature_names]
                    recommended = ['BC_Rate', 'BC_Temp', 'Metal_Available', 'BC_SSA']
                    default_feature = next((f for f in recommended if f in nf), nf[0])
                    df_feat = st.selectbox(
                        "选择特征",
                        nf,
                        index=nf.index(default_feature),
                        help="正文优先建议：BC_Rate、BC_Temp、Metal_Available、BC_SSA；Soil_Moisture建议先检查缺失率与数据来源结构。"
                    )
                    if st.button("生成SHAP Dependence"):
                        with st.spinner("计算中..."):
                            fig_dep = viz.plot_shap_dependence(
                                shap_v,
                                X_shap,
                                df_feat,
                                scaler=consultant.scaler,
                                feature_names=consultant.feature_names,
                                interaction_index=None
                            )
                            st.pyplot(fig_dep)
                # PDP
                st.divider();
                st.subheader("📉 偏依赖图 (PDP) 阈值分析")
                default_pdp = [f for f in ['BC_Temp', 'BC_Rate'] if f in consultant.feature_names]
                spdp = st.multiselect("选择PDP特征（建议1-2个）", consultant.feature_names, default=default_pdp,
                                      max_selections=2)
                if spdp:
                    with st.spinner("计算偏依赖..."):
                        fig_pdp = BiocharVisualizer.plot_pdp(consultant.model, consultant.X_train, spdp,
                                                             scaler=consultant.scaler,
                                                             feature_names=consultant.feature_names)
                        st.pyplot(fig_pdp)
                        st.caption(
                            "注：PDP阈值拐点为人工判读结果，建议描述为'在约XX附近呈现响应转折'，而非'算法自动识别最优值'。")
                # ============================================================
                # 污染物与场地条件驱动的修复响应差异
                # ============================================================
                st.divider()

                st.subheader(
                    "🌍 污染物与场地条件驱动的修复响应差异"
                )

                st.caption(
                    "Panels (a–c) 使用完整清洗数据库进行描述性场景分析；"
                    "Panel (d) 的subgroup SHAP仅基于训练集计算。"
                    "C_bio/C_0为施炭前有效态浓度与初始总浓度之比，用于描述性分组。模型使用初始有效态和总浓度原值。"
                )

                # ------------------------------------------------------------
                # 1. 构建描述性情景数据
                # ------------------------------------------------------------
                scenario_df, rbio_info = (
                    ScenarioResponseAnalyzer
                    .prepare_scenario_dataframe(df)
                )

                # ------------------------------------------------------------
                # 2. Metal_Type描述统计
                # ------------------------------------------------------------
                metal_order = (
                    ScenarioResponseAnalyzer
                    .get_metal_order(
                        scenario_df
                    )
                )

                metal_stats = (
                    ScenarioResponseAnalyzer
                    .descriptive_statistics(
                        scenario_df,
                        'Metal_Type',
                        group_order=metal_order
                    )
                )

                # ------------------------------------------------------------
                # 3. pH描述统计
                # ------------------------------------------------------------
                ph_stats = (
                    ScenarioResponseAnalyzer
                    .descriptive_statistics(
                        scenario_df,
                        'Soil_pH_Group',
                        group_order=
                        ScenarioResponseAnalyzer.PH_ORDER
                    )
                )

                # ------------------------------------------------------------
                # 4. R_bio描述统计
                # ------------------------------------------------------------
                rbio_stats = (
                    ScenarioResponseAnalyzer
                    .descriptive_statistics(
                        scenario_df,
                        'R_bio_Group',
                        group_order=[
                            'Low',
                            'Medium',
                            'High'
                        ]
                    )
                )

                # ------------------------------------------------------------
                # 5. 探索性组间统计
                # ------------------------------------------------------------
                metal_kw = (
                    ScenarioResponseAnalyzer
                    .kruskal_test(
                        scenario_df,
                        'Metal_Type'
                    )
                )

                ph_kw = (
                    ScenarioResponseAnalyzer
                    .kruskal_test(
                        scenario_df,
                        'Soil_pH_Group'
                    )
                )

                rbio_kw = (
                    ScenarioResponseAnalyzer
                    .kruskal_test(
                        scenario_df.dropna(
                            subset=['R_bio_Group']
                        ),
                        'R_bio_Group'
                    )
                )

                # ------------------------------------------------------------
                # 6. 控制subgroup最小样本量
                # ------------------------------------------------------------
                sc1, sc2 = st.columns(
                    [1, 2]
                )

                with sc1:

                    subgroup_min_n = st.slider(
                        "Subgroup SHAP最低样本量",
                        min_value=5,
                        max_value=30,
                        value=10,
                        step=1,
                        help=(
                            "论文建议至少10条。"
                            "样本数不足的Metal或pH组"
                            "不进入SHAP热图，"
                            "避免对小样本类别过度解释。"
                        ),
                        key="scenario_min_n"
                    )

                with sc2:

                    st.info(
                        "建议正式论文固定 min n = 10。"
                        "若某污染物类别少于10条，"
                        "仅报告描述性统计，"
                        "不解释其subgroup SHAP排序。"
                    )

                # ------------------------------------------------------------
                # 7. 构建sample-level grouped SHAP
                # ------------------------------------------------------------
                heatmap_df = pd.DataFrame()
                scenario_summary = pd.DataFrame()

                if (
                        shap_v is not None
                        and shap_v.ndim == 2
                        and consultant.raw_train_df is not None
                ):

                    try:

                        grouped_sample_shap = (
                            ScenarioResponseAnalyzer
                            .build_sample_level_grouped_shap(
                                shap_v,
                                consultant.feature_names
                            )
                        )

                        (
                            heatmap_df,
                            scenario_summary
                        ) = (
                            ScenarioResponseAnalyzer
                            .build_subgroup_shap(
                                consultant.raw_train_df,
                                grouped_sample_shap,
                                min_group_n=
                                subgroup_min_n,
                                top_n_features=10
                            )
                        )

                    except Exception as e:

                        st.warning(
                            "Subgroup SHAP计算失败："
                            f"{e}"
                        )

                else:

                    st.warning(
                        "当前模型没有可用的二维SHAP矩阵，"
                        "因此仅显示描述性场景分析。"
                    )

                # ============================================================
                # Fig. 6 FINAL PUBLICATION OUTPUT
                # 分图 + PNG/TIFF + 统一dpi
                # ============================================================
                st.divider()

                st.subheader(
                    "📊 Fig. 6 污染物与场地条件驱动的修复响应差异"
                )

                st.caption(
                    "四个子图分别生成。"
                )

                # ------------------------------------------------------------
                # 分辨率
                # ------------------------------------------------------------
                fig6_dpi = st.radio(
                    "导出分辨率",
                    options=[600, 900],
                    index=0,
                    horizontal=True,
                    key="fig6_export_dpi"
                )

                # ============================================================
                # Fig. 6a
                # ============================================================
                st.markdown(
                    "### Fig. 6a — Metal type"
                )

                fig6a = (
                    ScenarioResponseAnalyzer
                    .plot_figure6a_metal(
                        scenario_df
                    )
                )

                st.pyplot(
                    fig6a,
                    use_container_width=False
                )

                st.caption(
                    "Fig. 6a. Variation in immobilization efficiency "
                    "among different metal(loid) types."
                )

                ScenarioResponseAnalyzer.render_download_buttons(
                    fig=fig6a,
                    fig_no="6",
                    subtag="a",
                    title_key="MetalType_Response",
                    dpi=fig6_dpi,
                    key_prefix="fig6a",
                    cache=consultant.scenario_figure_exports
                )

                # 防止streamlit多次重绘占内存
                plt.close(fig6a)

                st.divider()

                # ============================================================
                # Fig. 6b
                # ============================================================
                st.markdown(
                    "### Fig. 6b — Soil pH"
                )

                fig6b = (
                    ScenarioResponseAnalyzer
                    .plot_figure6b_ph(
                        scenario_df
                    )
                )

                st.pyplot(
                    fig6b,
                    use_container_width=False
                )

                st.caption(
                    "Fig. 6b. Variation in immobilization efficiency "
                    "across initial soil-pH groups."
                )

                ScenarioResponseAnalyzer.render_download_buttons(
                    fig=fig6b,
                    fig_no="6",
                    subtag="b",
                    title_key="SoilpH_Response",
                    dpi=fig6_dpi,
                    key_prefix="fig6b",
                    cache=consultant.scenario_figure_exports
                )

                plt.close(fig6b)

                st.divider()

                # ============================================================
                # Fig. 6c
                # ============================================================
                st.markdown(
                    "### Fig. 6c — Initial bioavailable fraction"
                )

                fig6c = (
                    ScenarioResponseAnalyzer
                    .plot_figure6c_rbio(
                        scenario_df
                    )
                )

                st.pyplot(
                    fig6c,
                    use_container_width=False
                )

                st.caption(
                    "Fig. 6c. Variation in immobilization efficiency "
                    "across tertile-based groups of the initial "
                    "bioavailable fraction (C_bio/C_0)."
                )

                ScenarioResponseAnalyzer.render_download_buttons(
                    fig=fig6c,
                    fig_no="6",
                    subtag="c",
                    title_key="BioavailableFraction_Response",
                    dpi=fig6_dpi,
                    key_prefix="fig6c",
                    cache=consultant.scenario_figure_exports
                )

                plt.close(fig6c)

                st.divider()

                # ============================================================
                # Fig. 6d
                # ============================================================
                st.markdown(
                    "### Fig. 6d — Scenario-specific SHAP"
                )

                fig6d = (
                    ScenarioResponseAnalyzer
                    .plot_figure6d_heatmap(
                        heatmap_df
                    )
                )

                st.pyplot(
                    fig6d,
                    use_container_width=False
                )

                st.caption(
                    "Fig. 6d. Scenario-specific normalized SHAP "
                    "contributions. Values represent the within-scenario "
                    "relative contribution of each predictor."
                )

                ScenarioResponseAnalyzer.render_download_buttons(
                    fig=fig6d,
                    fig_no="6",
                    subtag="d",
                    title_key="SubgroupSHAP_Heatmap",
                    dpi=fig6_dpi,
                    key_prefix="fig6d",
                    cache=consultant.scenario_figure_exports,
                    options=subgroup_min_n
                )

                plt.close(fig6d)
                # ------------------------------------------------------------
                # 10. R_bio分组阈值
                # ------------------------------------------------------------
                st.markdown(
                    "#### C_bio/C_0 初始情景分组"
                )

                if (
                        np.isfinite(
                            rbio_info['q33']
                        )
                        and np.isfinite(
                    rbio_info['q67']
                )
                ):
                    rb1, rb2, rb3 = st.columns(3)

                    rb1.metric(
                        "有效R_bio样本数",
                        int(
                            rbio_info[
                                'n_valid'
                            ]
                        )
                    )

                    rb2.metric(
                        "33.3%分位数",
                        (
                            f"{100 * rbio_info['q33']:.2f}%"
                        )
                    )

                    rb3.metric(
                        "66.7%分位数",
                        (
                            f"{100 * rbio_info['q67']:.2f}%"
                        )
                    )

                    st.caption(
                        "Low / Medium / High按照当前数据库"
                        "C_bio/C_0的三分位数划分，"
                        "仅用于初始场景描述，"
                        "并非环境风险法规阈值。"
                    )

                # ------------------------------------------------------------
                # 11. 探索性Kruskal-Wallis结果
                # ------------------------------------------------------------
                st.markdown(
                    "#### 探索性组间差异检验"
                )

                kw1, kw2, kw3 = st.columns(3)

                with kw1:

                    if np.isfinite(
                            metal_kw['p']
                    ):
                        st.metric(
                            "Metal_Type",
                            (
                                f"H={metal_kw['H']:.2f}"
                            ),
                            (
                                f"p={metal_kw['p']:.4g}"
                            )
                        )
                    else:
                        st.metric(
                            "Metal_Type",
                            "N/A"
                        )

                with kw2:

                    if np.isfinite(
                            ph_kw['p']
                    ):
                        st.metric(
                            "Soil pH",
                            (
                                f"H={ph_kw['H']:.2f}"
                            ),
                            (
                                f"p={ph_kw['p']:.4g}"
                            )
                        )
                    else:
                        st.metric(
                            "Soil pH",
                            "N/A"
                        )

                with kw3:

                    if np.isfinite(
                            rbio_kw['p']
                    ):
                        st.metric(
                            "C_bio/C_0",
                            (
                                f"H={rbio_kw['H']:.2f}"
                            ),
                            (
                                f"p={rbio_kw['p']:.4g}"
                            )
                        )
                    else:
                        st.metric(
                            "C_bio/C_0",
                            "N/A"
                        )

                st.caption(
                    "Kruskal-Wallis检验仅作为探索性描述。"
                    "由于文献整合数据库并非随机平衡设计，"
                    "组间P值不能解释为Metal_Type、pH或"
                    "C_bio/C_0的独立因果效应。"
                )

                # ------------------------------------------------------------
                # 12. Supplementary Table S10
                # ------------------------------------------------------------
                st.markdown(
                    "#### Table S10 "
                    "各污染物与场地情景的描述性统计"
                )

                supplementary_s10 = []

                if not metal_stats.empty:
                    temp = metal_stats.copy()

                    temp.insert(
                        0,
                        'Grouping',
                        'Metal_Type'
                    )

                    supplementary_s10.append(
                        temp
                    )

                if not ph_stats.empty:
                    temp = ph_stats.copy()

                    temp.insert(
                        0,
                        'Grouping',
                        'Soil_pH'
                    )

                    supplementary_s10.append(
                        temp
                    )

                if not rbio_stats.empty:
                    temp = rbio_stats.copy()

                    temp.insert(
                        0,
                        'Grouping',
                        'C_bio/C_0'
                    )

                    supplementary_s10.append(
                        temp
                    )

                if len(
                        supplementary_s10
                ) > 0:
                    table_s10 = pd.concat(
                        supplementary_s10,
                        ignore_index=True
                    )

                    numeric_cols_s10 = [
                        c
                        for c in table_s10.columns
                        if c not in [
                            'Grouping',
                            'Group',
                            'N'
                        ]
                    ]

                    table_s10[
                        numeric_cols_s10
                    ] = table_s10[
                        numeric_cols_s10
                    ].round(2)

                    st.dataframe(
                        table_s10,
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇️ 下载 Table S10 CSV",
                        data=table_s10.to_csv(
                            index=False
                        ).encode(
                            'utf-8-sig'
                        ),
                        file_name=(
                            "Table_S10_"
                            "Scenario_Descriptive_Statistics.csv"
                        ),
                        mime="text/csv",
                        key="download_s10"
                    )

                # ------------------------------------------------------------
                # 13. Supplementary Table S11
                # normalized subgroup SHAP
                # ------------------------------------------------------------
                st.markdown(
                    "#### Table S11 "
                    "标准化Subgroup SHAP贡献"
                )

                if not heatmap_df.empty:

                    table_s11 = (
                        heatmap_df
                        .copy()
                        .round(2)
                    )

                    table_s11.insert(
                        0,
                        'N',
                        [
                            int(
                                scenario_summary.loc[
                                    scenario_summary[
                                        'Scenario'
                                    ] == idx,
                                    'N'
                                ].iloc[0]
                            )
                            if (
                                    scenario_summary[
                                        'Scenario'
                                    ] == idx
                            ).any()
                            else np.nan
                            for idx
                            in table_s11.index
                        ]
                    )

                    st.dataframe(
                        table_s11,
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇️ 下载 Table S11 CSV",
                        data=(
                            table_s11
                            .reset_index()
                            .rename(
                                columns={
                                    'index':
                                        'Scenario'
                                }
                            )
                            .to_csv(
                                index=False
                            )
                            .encode(
                                'utf-8-sig'
                            )
                        ),
                        file_name=(
                            "Table_S11_"
                            "Normalized_Subgroup_SHAP.csv"
                        ),
                        mime="text/csv",
                        key="download_s11"
                    )

                else:

                    st.info(
                        "当前没有满足最低样本量要求的"
                        "subgroup SHAP结果。"
                    )

                # ------------------------------------------------------------
                # 14. Table 4候选结果
                # ------------------------------------------------------------
                st.markdown(
                    "#### Table 4候选："
                    "场景特异性的关键变量与材料设计参数"
                )

                if not scenario_summary.empty:
                    scenario_display = (
                        scenario_summary
                        .copy()
                    )

                    scenario_display[
                        'Top_design_SHAP_%'
                    ] = (
                        scenario_display[
                            'Top_design_SHAP_%'
                        ].round(2)
                    )

                    st.dataframe(
                        scenario_display,
                        use_container_width=True
                    )

                    st.caption(
                        "Top_1–Top_3表示该场景中标准化SHAP"
                        "贡献最高的原始变量；"
                        "Top_design_variable仅在BC_Rate、"
                        "BC_Temp、BC_N和BC_SSA四个"
                        "可调设计变量中进行比较。"
                        "该表用于支撑向逆向设计的过渡，"
                        "不表示因果机制排序。"
                    )

                    st.download_button(
                        "⬇️ 下载 Table 4候选 CSV",
                        data=(
                            scenario_display
                            .to_csv(
                                index=False
                            )
                            .encode(
                                'utf-8-sig'
                            )
                        ),
                        file_name=(
                            "Table4_"
                            "Scenario_Dependent_"
                            "Design_Requirements.csv"
                        ),
                        mime="text/csv",
                        key="download_table4"
                    )

                # 学习曲线
                st.divider()
                st.subheader("🧪 模型诊断（Supplementary）")
                clc, _ = st.columns(2)
                with clc:
                    st.write("#### 学习曲线（诊断图）")
                    plotter = ScientificPlotter()
                    if st.button("生成学习曲线", key="generate_learning_curve"):
                        try:
                            consultant.learning_curve_data = plotter.compute_learning_curve(consultant)
                        except ValueError as exc:
                            st.warning(str(exc))
                    if consultant.learning_curve_data is not None:
                        learning_fig = plotter.plot_learning_curve(*consultant.learning_curve_data)
                        st.pyplot(learning_fig)
                        plt.close(learning_fig)
                    st.caption("各训练子集独立拟合插补、标准化和编码。图为固定超参数的诊断结果。")

                # 可行性校准
                st.divider()
                st.subheader("⚙️ 可行性评分体系校准")
                st.caption(
                    "按所选折数在训练数据上执行 OOF 校准，独立测试集不参与。"
                    "在非负单纯形上以step=0.02确定性网格搜索w_B、w_N、w_D，"
                    "选择使Spearman ρ(F_fea, e_OOF)最负的权重组合。仅当T_H > T_M时进行三级评级。"
                )

                cal1, cal2, cal3 = st.columns(3)
                with cal1:
                    n_splits = st.slider("交叉验证折数", 3, 10, 5, 1)
                with cal2:
                    error_high = st.number_input("可信组OOF误差阈值 (%)", 2.0, 20.0, 8.0, 0.5)
                with cal3:
                    error_medium = st.number_input("风险组OOF误差阈值 (%)", 5.0, 30.0, 15.0, 1.0)

                weight_step = 0.02
                st.caption(
                    "权重搜索固定步长：0.02。约束：w_B + w_N + w_D = 1，且各权重≥0。"
                    "其中w_N对应ECDF近邻支持（Neighborhood），w_D对应局部预测偏差（Deviation）。"
                )

                if st.button("🔧 执行OOF权重自动校准"):
                    with st.spinner(f"正在执行训练数据 {n_splits} 折 OOF 校准与单纯形权重搜索..."):
                        try:
                            consultant.calibration_output = consultant.run_feasibility_calibration(
                                n_splits=n_splits,
                                error_high=error_high,
                                error_medium=error_medium,
                                weight_step=weight_step
                            )
                        except ValueError as exc:
                            st.error(f"校准未完成：{exc}")

                if consultant.calibration_output is not None:
                    best_weights, best_thresholds, metrics = consultant.calibration_output
                    st.caption(f"以下结果对应最近一次成功校准（{metrics['cv_folds']} 折，"
                               f"误差分组阈值 {metrics['error_high']:g}% / {metrics['error_medium']:g}%）。")
                    if consultant.calibrator and consultant.calibrator.calibration_error:
                        st.markdown(
                            f'<div class="calib-error">❌ {consultant.calibrator.calibration_error}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.success("校准完成：仅使用训练数据OOF结果，独立测试集未参与。")

                        if consultant.calibrator and consultant.calibrator.threshold_warning:
                            st.markdown(
                                f'<div class="calib-warning">⚠️ {consultant.calibrator.threshold_warning}</div>',
                                unsafe_allow_html=True
                            )

                        # -------------------------------------------------
                        # 最优权重 + rho
                        # -------------------------------------------------
                        st.markdown("#### 🎯 自动搜索得到的最优权重")
                        rw1, rw2, rw3, rw4 = st.columns(4)
                        rw1.metric("w_B · Boundary", f"{best_weights['boundary']:.2f}")
                        rw2.metric("w_N · Neighborhood", f"{best_weights['distance']:.2f}")
                        rw3.metric("w_D · Deviation", f"{best_weights['deviation']:.2f}")
                        rw4.metric(
                            "Spearman ρ",
                            f"{metrics['spearman_correlation']:.4f}",
                            help="越负表示可行性得分越高时OOF误差总体越低。"
                        )

                        p_val = metrics.get('spearman_p_value', np.nan)
                        n_comb = metrics.get('weight_combinations_tested', None)
                        n_cal = metrics.get('calibration_n', None)
                        st.caption(
                            f"OOF有效校准样本 n={n_cal}; 搜索组合数={n_comb}; "
                            f"post-selection Spearman p="
                            f"{('< 0.0001' if np.isfinite(p_val) and p_val < 0.0001 else f'{p_val:.4g}') if np.isfinite(p_val) else 'NA'}。"
                            "由于权重在同一OOF数据上选择，该p值仅作探索性描述，不作为独立显著性证据。"
                        )

                        # -------------------------------------------------
                        # Top 10组合
                        # -------------------------------------------------
                        if consultant.calibrator.weight_search_results is not None:
                            st.markdown("#### 📋 Top 10 权重组合（Supplementary Table S14候选）")
                            top10 = consultant.calibrator.weight_search_results.head(10).copy()
                            display_cols = ['Rank', 'w_B', 'w_N', 'w_D', 'rho_s', 'p_value']
                            top10_show = top10[display_cols].copy()
                            for c in ['w_B', 'w_N', 'w_D']:
                                top10_show[c] = top10_show[c].round(2)
                            top10_show['rho_s'] = top10_show['rho_s'].round(4)
                            top10_show['p_value'] = top10_show['p_value'].apply(
                                lambda x: '<0.0001' if pd.notna(x) and x < 0.0001
                                else (f'{x:.4g}' if pd.notna(x) else 'NA')
                            )
                            st.dataframe(top10_show, use_container_width=True)

                            st.download_button(
                                "⬇️ 下载完整权重搜索结果 CSV",
                                data=consultant.calibrator.weight_search_results.to_csv(index=False).encode('utf-8-sig'),
                                file_name="Table_S14_Feasibility_Weight_Grid_Search.csv",
                                mime="text/csv",
                                key="download_feasibility_weight_grid"
                            )

                        # -------------------------------------------------
                        # 阈值结果
                        # -------------------------------------------------
                        st.markdown("#### 📐 OOF误差驱动的经验阈值")
                        if best_thresholds.get('separable', False):
                            rt1, rt2 = st.columns(2)
                            rt1.metric("T_H · High-support threshold", f"{best_thresholds['high']:.6g}")
                            rt2.metric("T_M · Medium/Low threshold", f"{best_thresholds['medium']:.6g}")
                        else:
                            st.info(
                                "当前可信组和风险组的可行性得分未形成稳定分离，因此不强制输出High/Medium/Low三级。"
                                "后续建议报告连续F_fea、Neighborhood score及风险来源。"
                            )

                        # -------------------------------------------------
                        # 校准效果
                        # -------------------------------------------------
                        st.markdown("#### 📊 校准效果评估")
                        ev1, ev2, ev3 = st.columns(3)
                        ev1.metric(
                            "Spearman ρ",
                            f"{metrics['spearman_correlation']:.4f}"
                        )
                        ev2.metric(
                            "P值（探索性）",
                            '< 0.0001' if np.isfinite(p_val) and p_val < 0.0001
                            else (f"{p_val:.4f}" if np.isfinite(p_val) else 'NA')
                        )
                        if metrics.get('separable', False):
                            ev3.metric("High组平均OOF误差", f"{metrics['high_level_mape']:.2f}%")
                            st.caption(
                                f"样本分布：High {metrics['high_level_count']} | "
                                f"Medium {metrics['mid_level_count']} | Low {metrics['low_level_count']}"
                            )
                        else:
                            ev3.metric("三级分级", "未建立")

                        st.markdown("#### 📈 可行性得分与OOF误差")
                        df_calib = consultant.calibrator.calibration_df.copy()
                        st.pyplot(
                            BiocharVisualizer.plot_feasibility_calibration(
                                df_calib,
                                best_weights,
                                best_thresholds,
                                metrics,
                                separable=best_thresholds.get('separable', False)
                            )
                        )

                        # -------------------------------------------------
                        #
                        # -------------------------------------------------
                        st.markdown("#### 📝 论文结果摘要")
                        st.code(
                            f"w_B={best_weights['boundary']:.2f}, "
                            f"w_N={best_weights['distance']:.2f}, "
                            f"w_D={best_weights['deviation']:.2f}; "
                            f"Spearman rho={metrics['spearman_correlation']:.4f}; "
                            f"p={p_val:.4g}" if np.isfinite(p_val) else
                            f"w_B={best_weights['boundary']:.2f}, "
                            f"w_N={best_weights['distance']:.2f}, "
                            f"w_D={best_weights['deviation']:.2f}; "
                            f"Spearman rho={metrics['spearman_correlation']:.4f}; p=NA",
                            language=None
                        )


    elif app_mode == "单样本快速预测":
        st.header("⚡ 单样本快速预测")
        if not consultant.is_trained:
            st.error("请先完成模型训练。")
        else:
            cat_cols = FEATURE_CONFIG['categorical']
            fc = consultant.onehot.categories_[cat_cols.index('BC_Feedstock')]
            mc = consultant.onehot.categories_[cat_cols.index('Metal_Type')]
            sc = consultant.onehot.categories_[cat_cols.index('Soil_Type')]
            with st.form("single_sample_form"):
                st.subheader("一、土壤基础属性")
                cs1, cs2, cs3 = st.columns(3)
                with cs1: soil_ph = st.number_input("土壤 pH", 3.0, 10.0, 6.5, 0.1); soil_clay = st.number_input(
                    "黏土含量 (%)", 0.0, 100.0, 20.0, 0.5)
                with cs2: soil_som = st.number_input("有机质 SOM (g/kg)", 0.0, 150.0, 25.0,
                                                     1.0); soil_cec = st.number_input("CEC (cmol/kg)", 0.0, 100.0, 15.0,
                                                                                      1.0)
                with cs3: soil_moisture = st.number_input("含水率 (%)", 5.0, 50.0, 25.0, 1.0); soil_type = st.selectbox(
                    "土壤类型", sc)
                st.divider();
                st.subheader("二、生物炭参数")
                cb1, cb2, cb3 = st.columns(3)
                with cb1: bc_feedstock = st.selectbox("生物炭原料", fc); bc_rate = st.number_input("施用量 (%)", 0.1,
                                                                                                   10.0, 2.0, 0.1)
                with cb2: bc_temp = st.number_input("热解温度 (℃)", 300, 800, 500, 10); bc_ssa = st.number_input(
                    "比表面积 SSA (m²/g)", 20.0, 1000.0, 150.0, 10.0)
                with cb3: bc_n = st.number_input("氮含量 (%)", 0.1, 15.0, 1.5, 0.1); bc_ash = st.number_input(
                    "灰分 (%)", 5.0, 50.0, 15.0, 1.0)
                st.divider();
                st.subheader("三、重金属参数")
                cm1, cm2 = st.columns(2)
                with cm1: metal_type = st.selectbox("重金属类型", mc); metal_total = st.number_input("初始总浓度 (mg/kg)",
                                                                                                     0.0, 3000.0, 200.0,
                                                                                                     10.0)
                with cm2: metal_available = st.number_input("初始有效态浓度 (mg/kg)", 0.0, 1500.0, 70.0, 5.0)
                submit_btn = st.form_submit_button("🔍 开始预测")
            if submit_btn:
                input_data = {'Soil_pH': soil_ph, 'Soil_Clay': soil_clay, 'Soil_SOM': soil_som,
                              'Soil_CEC': soil_cec, 'Soil_Moisture': soil_moisture, 'Soil_Type': soil_type,
                              'BC_Feedstock': bc_feedstock, 'BC_Rate': bc_rate, 'BC_Temp': bc_temp,
                              'BC_SSA': bc_ssa, 'BC_N': bc_n, 'BC_Ash': bc_ash,
                              'Metal_Type': metal_type, 'Metal_Total': metal_total, 'Metal_Available': metal_available}
                try:
                    val_result = consultant.validate_formulation(input_data)
                    predicted_ie = val_result['predicted_ie']
                    st.success("预测完成！");
                    st.markdown("### 🎯 预测结果")
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("固定效率 (IE)", f"{predicted_ie:.2f} %")
                    level = "高修复效果" if predicted_ie > 70 else "中等修复效果" if predicted_ie > 40 else "低修复效果"
                    rc2.metric("效果等级", level)
                    rc3.metric("可行性得分", f"{val_result['total_score']:.1f} 分")
                    st.info(f"可行性评价：{val_result['level']} | {val_result['risk_tip']}")
                    history_mgr.save_record({'ph': soil_ph, 'metal': metal_type, 'rate': bc_rate,
                                             'temp': bc_temp, 'ie': predicted_ie, 'score': predicted_ie / 100,
                                             'feasibility': val_result['total_score']})
                    st.toast("预测结果已存入历史记录", icon="💾")
                except ValueError as exc:
                    st.error(f"预测未完成：{exc}")
    elif app_mode == "AI反向配方定制":
        st.header("🤖 AI 反向材料定制（差分进化加权多目标优化）")
        if not consultant.is_trained:
            st.error("请先完成模型训练。")
        else:
            cat_cols = FEATURE_CONFIG['categorical']
            fc = consultant.onehot.categories_[cat_cols.index('BC_Feedstock')]
            mc = consultant.onehot.categories_[cat_cols.index('Metal_Type')]
            sc = consultant.onehot.categories_[cat_cols.index('Soil_Type')]
            st.sidebar.markdown("### ⚖️ 多目标偏好（自动归一化）")
            w_ie = st.sidebar.slider("固定率 IE", 0.0, 1.0, 0.7)
            w_cost = st.sidebar.slider("相对成本代理", 0.0, 1.0, 0.2)
            w_fert = st.sidebar.slider("肥力相关代理", 0.0, 1.0, 0.1)
            ws = w_ie + w_cost + w_fert
            if ws > 0: st.sidebar.caption(f"归一化: IE={w_ie / ws:.2f}, Cost={w_cost / ws:.2f}, Fert={w_fert / ws:.2f}")
            st.write("#### 第一步：输入污染场地特征")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                ph = st.number_input("土壤 pH", 3.0, 10.0, 6.5); clay = st.number_input("黏土 (%)", 0.0, 100.0, 20.0)
            with s2:
                som = st.number_input("有机质 (g/kg)", 0.0, 150.0, 25.0); cec = st.number_input("CEC", 0.0, 100.0, 15.0)
            with s3:
                m_total = st.number_input("初始总浓度 (mg/kg)", 0.0, 3000.0, 200.0); m_type = st.selectbox("重金属", mc)
            with s4:
                s_type = st.selectbox("土壤类型", sc); moisture = st.number_input("含水率 (%)", 5.0, 50.0, 25.0)
            st.write("#### 第二步：AI 寻优")
            target_feedstock = st.selectbox("预选生物炭来源", fc)
            default_available = st.checkbox("用总量的 35% 作为初始有效态情景值", value=True)
            available = st.number_input(
                "实测初始有效态浓度 (mg/kg)", min_value=0.0, max_value=100000.0,
                value=70.0, disabled=default_available
            )
            default_ash = st.checkbox("用 15% 作为灰分情景值", value=True)
            ash = st.number_input(
                "实测灰分 (%)", min_value=0.0, max_value=100.0, value=15.0, disabled=default_ash
            )
            if st.button("🪄 生成最佳定制方案"):
                soil_input = {'Soil_pH': ph, 'Soil_Clay': clay, 'Soil_SOM': som, 'Soil_CEC': cec,
                              'Soil_Moisture': moisture, 'Metal_Total': m_total,
                              'Metal_Available': m_total * 0.35 if default_available else available,
                              'BC_Ash': 15.0 if default_ash else ash,
                              'Metal_Type': m_type, 'Soil_Type': s_type, 'BC_Feedstock': target_feedstock}
                weights = {'ie': w_ie, 'cost': w_cost, 'fert': w_fert}
                try:
                    with st.spinner("🚀 差分进化引擎正在搜索加权折衷最优方案..."):
                        baseline_input = {
                            **soil_input, 'BC_Rate': 2.0, 'BC_Temp': 500.0, 'BC_N': 1.5, 'BC_SSA': 150.0
                        }
                        baseline_df, _ = consultant.validate_prediction_inputs(pd.DataFrame([baseline_input]))
                        baseline_proc = consultant.preprocess(baseline_df, is_training=False)
                        baseline_ie = consultant.model.predict(baseline_proc)[0]
                        opt_res = AdvancedOptimizer(consultant).run_optimization(soil_input, weights)
                        if opt_res.success:
                            st.success("✅ 已生成有近邻数据支持的候选配方。")
                            st.caption(
                                f"参考范围：{opt_res.support_scope}，共 {opt_res.support_count} 组观测配方组合。"
                                f"{'候选取自观测配方。' if opt_res.used_observed_recipe else '候选由差分进化搜索得到。'}"
                                f"初始有效态 {soil_input['Metal_Available']:.2f} mg/kg"
                                f"（{'情景值' if default_available else '实测输入'}），"
                                f"灰分 {soil_input['BC_Ash']:.2f}%（{'情景值' if default_ash else '实测输入'}）。"
                            )
                            best_rate, best_temp, best_n, best_ssa = opt_res.x
                            best_input = {**soil_input, 'BC_Rate': best_rate, 'BC_Temp': best_temp,
                                          'BC_N': best_n, 'BC_SSA': best_ssa}
                            val_result = consultant.validate_formulation(best_input)
                            predicted_ie = val_result['predicted_ie']
                            st.markdown("### 🎯 修复效率对比")
                            cp1, cp2 = st.columns(2)
                            cp1.metric("参考配方预测值", f"{baseline_ie:.2f}%")
                            cp1.caption("施用量 2%，热解温度 500 ℃，氮 1.5%，比表面积 150 m²/g。")
                            cp2.metric("AI定制生物炭预估", f"{predicted_ie:.2f}%",
                                       delta=f"{predicted_ie - baseline_ie:+.2f} 个百分点")
                            st.write("---");
                            st.markdown("### 📋 推荐配方")
                            rs1, rs2, rs3, rs4 = st.columns(4)
                            rs1.metric("热解温度", f"{best_temp:.0f} °C", f"{best_temp - 500:+.0f} °C")
                            rs2.metric("氮含量", f"{best_n:.2f} %", f"{best_n - 1.5:+.2f} %")
                            rs3.metric("施用量", f"{best_rate:.2f} %", f"{best_rate - 2.0:+.2f} %")
                            rs4.metric("比表面积", f"{best_ssa:.0f} m²/g", f"{best_ssa - 150:+.0f}")
                            st.divider();
                            st.markdown("### ✅ 配方可行性验证")
                            vc1, vc2, vc3 = st.columns(3)
                            vc1.metric("综合可行性得分", f"{val_result['total_score']} 分")
                            vc2.metric("可行性等级", val_result['level'])
                            vc3.metric("与相似样本偏差", f"{val_result['relative_deviation']} %")
                            if "高可行性" in val_result['level']:
                                st.success(f"💡 {val_result['risk_tip']}")
                            elif "中等可行性" in val_result['level']:
                                st.warning(f"⚠️ {val_result['risk_tip']}")
                            elif "低可行性" in val_result['level']:
                                st.error(f"🚨 {val_result['risk_tip']}")
                            else:
                                st.info(f"ℹ️ {val_result['risk_tip']}")
                            with st.expander("📏 边界合规明细", expanded=False):
                                bdf = pd.DataFrame(val_result['boundary_check']).T
                                bdf.columns = ['推荐值', '训练集5%', '训练集95%', '是否合规']
                                bdf['是否合规'] = bdf['是否合规'].map({True: '✅', False: '❌'})
                                st.dataframe(bdf, use_container_width=True)
                            with st.expander("🔬 相似训练样本", expanded=False):
                                dcols = [c for c in FEATURE_CONFIG['numerical'] + FEATURE_CONFIG['categorical'] + [
                                    FEATURE_CONFIG['target'], 'Study_ID', 'Source_Title', '特征距离'] if c in val_result['similar_samples'].columns]
                                st.dataframe(val_result['similar_samples'][dcols], use_container_width=True)
                            st.divider();
                            st.markdown("### 🛠️ 个性化工艺建议（rule-based，经验规则）")
                            advice = consultant.generate_process_advice(best_rate, best_temp, best_n, best_ssa,
                                                                        target_feedstock, m_type, ph, val_result)
                            ac1, ac2 = st.columns(2)
                            with ac1:
                                st.markdown(f'<div class="advice-card"><b>🔥 热解工艺</b><br>{advice["pyrolysis"]}</div>',
                                            unsafe_allow_html=True)
                                st.markdown(f'<div class="advice-card"><b>🌱 施用工艺</b><br>{advice["application"]}</div>',
                                            unsafe_allow_html=True)
                            with ac2:
                                st.markdown(f'<div class="advice-card"><b>🎯 靶向改性</b><br>{advice["metal_target"]}</div>',
                                            unsafe_allow_html=True)
                                st.markdown(
                                    f'<div class="advice-card"><b>📦 原料适配与风险</b><br>{advice["feedstock"]}<br><br>{advice["risk"]}</div>',
                                    unsafe_allow_html=True)
                            history_mgr.save_record({'ph': ph, 'metal': m_type, 'rate': best_rate, 'temp': best_temp,
                                                     'ie': predicted_ie, 'score': -opt_res.fun,
                                                     'feasibility': val_result['total_score']})
                            st.toast("方案已存入数据库", icon="💾")
                        else:
                            st.error("寻优引擎未能找到稳定解。")
                except ValueError as exc:
                    st.error(f"寻优未完成：{exc}")
    elif app_mode == "批量预测工具":
        st.header("📦 批量预测与导出")
        if not consultant.is_trained:
            st.warning("请先训练模型。")
        else:
            st.info("上传包含全部输入特征的CSV，系统自动补齐预测结果。")
            batch_file = st.file_uploader("上传批量数据", type="csv", key="batch")
            if batch_file:
                try:
                    b_df = pd.read_csv(batch_file, encoding='utf-8-sig')
                    clean_batch, logs = consultant.validate_prediction_inputs(b_df)
                    for log in logs:
                        st.info(log)
                    b_proc = consultant.preprocess(clean_batch, is_training=False)
                    b_df['Predicted_IE_Percent'] = np.round(consultant.model.predict(b_proc), 2)
                    st.dataframe(b_df)
                    output = io.BytesIO()
                    b_df.to_csv(output, index=False)
                    st.download_button("📩 下载预测报告", output.getvalue(), "biochar_batch_report.csv", "text/csv")
                except ValueError as exc:
                    st.error(f"批量预测未完成：{exc}")
    elif app_mode == "历史记录查询":
        st.header("📜 历史记录查询")
        stats = history_mgr.get_statistics()
        if stats is None:
            st.info("暂无历史记录。")
        else:
            st.subheader("📊 统计概览")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("总预测次数", int(stats['total']))
            sc2.metric("平均固定效率", f"{stats['avg_ie']:.2f} %")
            sc3.metric("最高固定效率", f"{stats['max_ie']:.2f} %")
            sc4.metric("平均可行性得分", f"{stats['avg_feasibility']:.1f} 分")
            st.divider()
            st.subheader("📋 详细记录（最近100条）")
            hdf = history_mgr.get_recent_history(100)
            hdf = hdf.rename(columns={'id': '序号', 'timestamp': '预测时间', 'soil_ph': '土壤pH',
                                      'metal_type': '重金属类型', 'bc_rate': '施用量(%)', 'bc_temp': '热解温度(℃)',
                                      'predicted_ie': '预测固定效率(%)', 'expert_score': '综合得分',
                                      'feasibility_score': '可行性得分'})
            st.dataframe(hdf, use_container_width=True)
            st.divider()
            cc1, cc2 = st.columns([1, 3])
            with cc1:
                confirm = st.checkbox("确认清空")
            with cc2:
                if confirm and st.button("🗑️ 清空全部", type="primary"):
                    history_mgr.clear_all()
                    st.success("已清空")
                    st.rerun()


if __name__ == "__main__":
    main()

