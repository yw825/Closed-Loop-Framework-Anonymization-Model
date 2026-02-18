import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, shapiro
from sklearn.model_selection import GridSearchCV
import gc
import itertools
from sklearn.utils import resample
import ast
import json
import re
import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from constants import *
import particle_swarm

###################################################################################################################################

# Data Preparation 

def data_prep(df):

    # Convert columns with object dtype to category dtype
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype('category')

    # df = df.set_index('PatientIdentifier')
   
    return df

def encode_categorical_from_file(df):
    
    # Apply one-hot encoding to categorical columns
    df_encoded = pd.get_dummies(df, drop_first=True)  # drop_first=True avoids dummy variable trap
    
    return df_encoded


###################################################################################################################################

# Closed_loop framework calcuations

def calculate_nqi_cqi_stats(df, NQIs, CQIs):

    stats = {}

    # Compute mean for NQIs
    for nqi in NQIs:
        if nqi in df.columns:
            stats[nqi] = df[nqi].mean()
    
    # Compute mode for CQIs
    for cqi in CQIs:
        if cqi in df.columns:
            stats[cqi] = df[cqi].mode()[0]  # mode() returns a Series, take first mode

    return stats



def calculate_denominator(df, NQIs, CQIs):

    # For NQIs, use NumPy for vectorized calculation of squared deviations from the mean
    denominator_nqi = sum(np.sum((df[nqi].values - df[nqi].mean())**2) for nqi in NQIs if nqi in df.columns)
    
    # For CQIs, use NumPy to calculate how many different values exist compared to the mode
    denominator_cqi = 0
    for cqi in CQIs:
        if cqi in df.columns:
            mode_cqi = df[cqi].mode()[0]  # Get the most frequent category for the CQI
            # Use NumPy to count non-mode values
            denominator_cqi += np.sum(df[cqi].values != mode_cqi)
    
    # Total denominator is the sum of the variances for NQIs and deviations for CQIs
    denominator = denominator_nqi + denominator_cqi
    return denominator



def get_cqi_levels(df, CQIs):

    levels_dict = {}
    for cqi in CQIs:
        if cqi in df.columns:
            # Map each unique category in the CQI column to a unique index
            levels_dict[cqi] = {category: index for index, category in enumerate(df[cqi].unique())}
    
    return levels_dict



def get_nqi_bounds(df, NQIs):

    bounds = {}
    
    for nqi in NQIs:
        if nqi in df.columns:
            lower_bound = df[nqi].min()
            # upper_bound = df[nqi].max()
            upper_bound = df[nqi].quantile(0.95)
            bounds[nqi] = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
    
    return bounds



def calculate_information_loss(original_df, anonymized_df, NQIs, CQIs, denominator):

    # Sort original data and anonymized data by 'PatientIdentifier'
    original_df = original_df.sort_index()
    anonymized_df = anonymized_df.sort_index()
    
    # Compute squared differences for numerical quasi-identifiers
    num_loss = np.sum((original_df[NQIs].values - anonymized_df[NQIs].values) ** 2)

    # Compute categorical mismatch (0 if same, 1 if different)
    cat_loss = np.sum(original_df[CQIs].values != anonymized_df[CQIs].values)

    # Total information loss per record
    total_loss = num_loss + cat_loss

    infoloss=total_loss/denominator

    return infoloss


def normalize_data(value, min_value, max_value):

    return (value - min_value) / (max_value - min_value + 1e-6)


###################################################################################################################################

# Privacy Attack Metrics

def get_linkage_attack(df, k=20):
    total = 0
    grouped_df = df.groupby(['cluster'])

    for name, group in grouped_df:
        if len(group) < k:
            total += len(group)
    return total

def get_homogeneity_attack(
    df,
    SAs,
    evaluate_joint=False
):
    """
    Parameters
    ----------
    df : DataFrame
        Anonymized data with 'cluster' column.
    SAs : list
        List of sensitive attribute column names.
    evaluate_joint : bool, default=False
        Whether to also evaluate homogeneity on joint SAs.

    Returns
    -------
    results : dict
        {
            "per_sa": { SA_name: num_individuals_at_risk },
            "joint": num_individuals_at_risk (or None)
        }
    """
    grouped_df = df.groupby("cluster")

    results = {
        "per_sa": {},
        "joint": None
    }

    # -------------------------
    # Per-SA homogeneity
    # -------------------------
    for SA in SAs:
        vulnerable_indices = set()

        for _, group in grouped_df:
            if group[SA].nunique() == 1:
                vulnerable_indices.update(group.index)

        results["per_sa"][SA] = len(vulnerable_indices)

    # -------------------------
    # Joint-SA homogeneity (optional)
    # -------------------------
    if evaluate_joint and len(SAs) > 1:
        vulnerable_indices = set()

        for _, group in grouped_df:
            if group[SAs].apply(tuple, axis=1).nunique() == 1:
                vulnerable_indices.update(group.index)

        results["joint"] = len(vulnerable_indices)

    return results

def get_skewness_attack(
    df,
    SAs,
    threshold,
    evaluate_joint=False
):
    """
    Parameters
    ----------
    df : DataFrame
        Anonymized data with 'cluster' column.
    SAs : list
        List of sensitive attribute column names.
    threshold : float
        Confidence threshold τ (e.g., 0.4).
    evaluate_joint : bool, default=False
        Whether to evaluate skewness on joint SAs.

    Returns
    -------
    results : dict
        {
            "per_sa": { SA_name: num_individuals_at_risk },
            "joint": num_individuals_at_risk (or None)
        }
    """
    grouped_df = df.groupby("cluster")

    results = {
        "per_sa": {},
        "joint": None
    }

    # -------------------------
    # Per-SA skewness
    # -------------------------
    for SA in SAs:
        vulnerable_indices = set()

        for _, group in grouped_df:
            value_dist = group[SA].value_counts(normalize=True)
            risky_values = value_dist[value_dist > threshold].index

            if len(risky_values) > 0:
                idx = group[group[SA].isin(risky_values)].index
                vulnerable_indices.update(idx)

        results["per_sa"][SA] = len(vulnerable_indices)

    # -------------------------
    # Joint-SA skewness (optional)
    # -------------------------
    if evaluate_joint and len(SAs) > 1:
        vulnerable_indices = set()

        for _, group in grouped_df:
            combined = group[SAs].apply(tuple, axis=1)
            value_dist = combined.value_counts(normalize=True)
            risky_values = value_dist[value_dist > threshold].index

            if len(risky_values) > 0:
                idx = group[combined.isin(risky_values)].index
                vulnerable_indices.update(idx)

        results["joint"] = len(vulnerable_indices)

    return results
    

def evaluate_privacy_attacks(
    df,
    SAs,
    k,
    skew_threshold,
    evaluate_joint=False
):
    results = []

    # -------------------------
    # Linkage attack
    # -------------------------
    linkage_count = get_linkage_attack(df, k)

    results.append({
        "Attack_Type": "Linkage",
        "SA": "N/A",
        "Num_Individuals_At_Risk": linkage_count
    })

    # -------------------------
    # Homogeneity attack
    # -------------------------
    homogeneity_results = get_homogeneity_attack(df, SAs)

    for sa, count in homogeneity_results["per_sa"].items():
        results.append({
            "Attack_Type": "Homogeneity",
            "SA": sa,
            "Num_Individuals_At_Risk": count
        })

    if evaluate_joint and homogeneity_results["joint"] is not None:
        results.append({
            "Attack_Type": "Homogeneity",
            "SA": "JOINT",
            "Num_Individuals_At_Risk": homogeneity_results["joint"]
        })

    # -------------------------
    # Skewness attack
    # -------------------------
    skew_results = get_skewness_attack(
        df,
        SAs,
        threshold=skew_threshold
    )

    for sa, count in skew_results["per_sa"].items():
        results.append({
            "Attack_Type": "Skewness",
            "SA": sa,
            "Num_Individuals_At_Risk": count
        })

    if evaluate_joint and skew_results["joint"] is not None:
        results.append({
            "Attack_Type": "Skewness",
            "SA": "JOINT",
            "Num_Individuals_At_Risk": skew_results["joint"]
        })

    return results


def parse_anonymized_filename(filename, dataset, process_type):

    info = {
        "Dataset": dataset,
        "ML_model": "N/A",
        "k": None,
        "n_clusters": None,
        "round": None
    }

    # -----------------------------
    # CLOSED LOOP
    # best_German_credit_DT_k20_ncluster20_round1.csv
    # -----------------------------
    if process_type.startswith("CLOSED_LOOP"):

        pattern = (
            r"best_.*?_"
            r"(?P<ML_model>[^_]+)_"
            r"k(?P<k>\d+)_"
            r"ncluster(?P<n_clusters>\d+)_"
            r"round(?P<round>\d+)"
        )

        m = re.search(pattern, filename)
        if not m:
            raise ValueError(f"Cannot parse closed-loop filename: {filename}")

        info.update({
            "ML_model": m.group("ML_model"),
            "k": int(m.group("k")),
            "n_clusters": int(m.group("n_clusters")),
            "round": int(m.group("round"))
        })

    # -----------------------------
    # MO-OBAM
    # best_anonymized_df_k20_ncluster10_round0.csv
    # -----------------------------
    elif process_type == "MO-OBAM":

        pattern = (
            r"best_anonymized_df_"
            r"k(?P<k>\d+)_"
            r"ncluster(?P<n_clusters>\d+)_"
            r"round(?P<round>\d+)"
        )

        m = re.search(pattern, filename)
        if not m:
            raise ValueError(f"Cannot parse MO-OBAM filename: {filename}")

        info.update({
            "k": int(m.group("k")),
            "n_clusters": int(m.group("n_clusters")),
            "round": int(m.group("round"))
        })

    # -----------------------------
    # BASELINE
    # -----------------------------
    elif process_type == "BASELINE":
        info.update({
            "round": 0
        })

    else:
        raise ValueError(f"Unknown process type: {process_type}")

    return info

def get_anonymized_files(dataset, process_type):
    """
    Returns a list of dicts with:
    Dataset | ML_model | k | n_clusters | round | path
    """

    source = EXPERIMENT_SOURCES[process_type]
    files = []

    # =========================
    # BASELINE
    # =========================
    if process_type == "BASELINE":
        return [{
            "path": DATASET_CONFIGS[dataset]["path"],
            "ML_model": "N/A",
            "k": None,
            "n_clusters": None,
            "round": 0
        }]

    base = source["base_path"]

    # =========================
    # MO-OBAM (no ML subfolder)
    # =========================
    if process_type == "MO-OBAM":
        data_dir = os.path.join(base, dataset, "Anonymized Data")

        for fname in os.listdir(data_dir):
            if not fname.endswith(".csv"):
                continue

            info = parse_anonymized_filename(
                filename=fname,
                dataset=dataset,
                process_type=process_type
            )
            info["path"] = os.path.join(data_dir, fname)
            info["ML_model"] = "N/A"
            files.append(info)

        return files

    # =========================
    # CLOSED LOOP (ML subfolders!)
    # =========================
    data_root = os.path.join(base, dataset, "Anonymized Data")

    for ml_model in os.listdir(data_root):
        ml_dir = os.path.join(data_root, ml_model)

        if not os.path.isdir(ml_dir):
            continue

        for fname in os.listdir(ml_dir):
            if not fname.endswith(".csv"):
                continue

            info = parse_anonymized_filename(
                filename=fname,
                dataset=dataset,
                process_type=process_type
            )

            info["path"] = os.path.join(ml_dir, fname)
            info["ML_model"] = ml_model
            files.append(info)

    return files

def add_baseline_clusters(df, QIs):

    df = df.copy()

    df["cluster"] = df.groupby(QIs, dropna=False).ngroup()

    return df


def plot_privacy_attack_vs_clusters(
    df_agg,
    dataset,
    ml_model,
    attack_type,
    sa,
    output_dir
):
    """
    Plot privacy attack results vs number of clusters.
    """

    # ---- Visual parameters ----
    FIG_SIZE = (120, 80)
    LINE_WIDTH = 80
    MARKER_SIZE = 150
    LABEL_SIZE = 210
    TICK_SIZE = 210
    LEGEND_SIZE = 210

    PROCESS_STYLE = {
        "CLOSED_LOOP_WITH_REPAIR": {"color": "blue", "linestyle": "-", "marker": "o"},
        "MO-OBAM": {"color": "green", "linestyle": "-", "marker": "s"},
        "CLOSED_LOOP_WITHOUT_REPAIR": {"color": "orange", "linestyle": "-", "marker": "^"}
    }

    PROCESS_LABEL = {
        "BASELINE": "Baseline",
        "CLOSED_LOOP_WITH_REPAIR": "Closed-Loop (With Repair)",
        "CLOSED_LOOP_WITHOUT_REPAIR": "Closed-Loop (No Repair)",
        "MO-OBAM": "MO-OBAM",
    }

    # --------------------------------------------------
    # Base filter: dataset + attack type + SA
    # --------------------------------------------------
    df_base = df_agg[
        (df_agg["Dataset"] == dataset) &
        (df_agg["Attack_Type"] == attack_type)
    ]

    if sa is None:
        df_base = df_base[df_base["SA"].isna()]
    else:
        df_base = df_base[df_base["SA"] == sa]

    if df_base.empty:
        return

    plt.figure(figsize=FIG_SIZE)
    legend_handles = []

    # --------------------------------------------------
    # Baseline (single value, legend only)
    # --------------------------------------------------
    baseline_df = df_base[df_base["Process_Type"] == "BASELINE"]

    if not baseline_df.empty:
        baseline_value = int(baseline_df["mean"].iloc[0])

        legend_handles.append(
            Line2D(
                [], [],
                label=f"Baseline = {baseline_value}"
            )
        )

    # --------------------------------------------------
    # MO-OBAM (NOT ML-dependent)
    # --------------------------------------------------
    mo_df = df_base[df_base["Process_Type"] == "MO-OBAM"]

    if not mo_df.empty:
        mo_df = mo_df.sort_values("n_clusters")

        plt.plot(
            mo_df["n_clusters"],
            mo_df["mean"],
            color=PROCESS_STYLE["MO-OBAM"]["color"],
            linestyle=PROCESS_STYLE["MO-OBAM"]["linestyle"],
            marker=PROCESS_STYLE["MO-OBAM"]["marker"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                color=PROCESS_STYLE["MO-OBAM"]["color"],
                linestyle=PROCESS_STYLE["MO-OBAM"]["linestyle"],
                marker=PROCESS_STYLE["MO-OBAM"]["marker"],
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label=PROCESS_LABEL["MO-OBAM"]
            )
        )

    # --------------------------------------------------
    # Closed-loop (ML-dependent)
    # --------------------------------------------------
    for process in [
        "CLOSED_LOOP_WITH_REPAIR",
        "CLOSED_LOOP_WITHOUT_REPAIR"
    ]:
        df_p = df_base[
            (df_base["Process_Type"] == process) &
            (df_base["ML_model"] == ml_model)
        ]

        if df_p.empty:
            continue

        df_p = df_p.sort_values("n_clusters")

        plt.plot(
            df_p["n_clusters"],
            df_p["mean"],
            color=PROCESS_STYLE[process]["color"],
            linestyle=PROCESS_STYLE[process]["linestyle"],
            marker=PROCESS_STYLE[process]["marker"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                color=PROCESS_STYLE[process]["color"],
                linestyle=PROCESS_STYLE[process]["linestyle"],
                marker=PROCESS_STYLE[process]["marker"],
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label=PROCESS_LABEL[process]
            )
        )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------
    plt.xlabel("Number of Clusters", fontsize=LABEL_SIZE)
    plt.ylabel("Number of Individuals at Risk", fontsize=LABEL_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)

    # --------------------------------------------------
    # Legend (top-center, 2 columns)
    # --------------------------------------------------
    plt.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=LEGEND_SIZE,
        frameon=False
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    sa_name = "None" if sa is None else sa
    fname = f"{dataset}_{ml_model}_{attack_type}_{sa_name}.png"
    out_path = os.path.join(output_dir, fname)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    plt.close()

def plot_privacy_attack_vs_clusters_worst_case(
    df_wc,
    dataset,
    ml_model,
    attack_type,
    output_dir
):
    """
    Plot WORST-CASE privacy attack results vs number of clusters.
    Assumes data is already aggregated across all SAs.
    """

    # ---- Visual parameters ----
    FIG_SIZE = (150, 80)
    LINE_WIDTH = 80
    MARKER_SIZE = 150
    LABEL_SIZE = 210
    TICK_SIZE = 210
    LEGEND_SIZE = 210

    PROCESS_STYLE = {
        "CLOSED_LOOP_WITH_REPAIR": {"color": "blue", "linestyle": "-", "marker": "o"},
        "MO-OBAM": {"color": "green", "linestyle": "-", "marker": "s"},
        "CLOSED_LOOP_WITHOUT_REPAIR": {"color": "orange", "linestyle": "-", "marker": "^"}
    }

    PROCESS_LABEL = {
        "BASELINE": "Baseline",
        "CLOSED_LOOP_WITH_REPAIR": "Closed-Loop (With Repair)",
        "CLOSED_LOOP_WITHOUT_REPAIR": "Closed-Loop (No Repair)",
        "MO-OBAM": "MO-OBAM",
    }

    # --------------------------------------------------
    # Base filter (NO SA here)
    # --------------------------------------------------
    df_base = df_wc[
        (df_wc["Dataset"] == dataset) &
        (df_wc["Attack_Type"] == attack_type)
    ]

    if df_base.empty:
        return
    
    # --------------------------------------------------
    # Collect ALL n_clusters values for x-axis
    # --------------------------------------------------
    x_values = (
        df_base["n_clusters"]
        .dropna()
        .unique()
    )
    x_values = sorted(x_values)

    plt.figure(figsize=FIG_SIZE)
    legend_handles = []

    # --------------------------------------------------
    # Baseline (single worst-case value)
    # --------------------------------------------------
    baseline_df = df_base[df_base["Process_Type"] == "BASELINE"]

    if not baseline_df.empty:
        baseline_value = int(baseline_df["Individuals_At_Risk_WC"].iloc[0])

        legend_handles.append(
            Line2D(
                [], [],
                label=f"Baseline = {baseline_value}"
            )
        )

    # --------------------------------------------------
    # MO-OBAM (NOT ML-dependent)
    # --------------------------------------------------
    mo_df = df_base[df_base["Process_Type"] == "MO-OBAM"]

    if not mo_df.empty:
        mo_df = mo_df.sort_values("n_clusters")

        plt.plot(
            mo_df["n_clusters"],
            mo_df["Individuals_At_Risk_WC"],
            color=PROCESS_STYLE["MO-OBAM"]["color"],
            linestyle=PROCESS_STYLE["MO-OBAM"]["linestyle"],
            marker=PROCESS_STYLE["MO-OBAM"]["marker"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                color=PROCESS_STYLE["MO-OBAM"]["color"],
                linestyle=PROCESS_STYLE["MO-OBAM"]["linestyle"],
                marker=PROCESS_STYLE["MO-OBAM"]["marker"],
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label=PROCESS_LABEL["MO-OBAM"]
            )
        )

    # --------------------------------------------------
    # Closed-loop (ML-dependent)
    # --------------------------------------------------
    for process in [
        "CLOSED_LOOP_WITH_REPAIR",
        "CLOSED_LOOP_WITHOUT_REPAIR"
    ]:
        df_p = df_base[
            (df_base["Process_Type"] == process) &
            (df_base["ML_model"] == ml_model)
        ]

        if df_p.empty:
            continue

        df_p = df_p.sort_values("n_clusters")

        plt.plot(
            df_p["n_clusters"],
            df_p["Individuals_At_Risk_WC"],
            color=PROCESS_STYLE[process]["color"],
            linestyle=PROCESS_STYLE[process]["linestyle"],
            marker=PROCESS_STYLE[process]["marker"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                color=PROCESS_STYLE[process]["color"],
                linestyle=PROCESS_STYLE[process]["linestyle"],
                marker=PROCESS_STYLE[process]["marker"],
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label=PROCESS_LABEL[process]
            )
        )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------
    plt.xlabel("Number of Clusters", fontsize=LABEL_SIZE)
    plt.ylabel("Worst-case Number of Individuals at Risk", fontsize=LABEL_SIZE)
    plt.xticks(x_values, fontsize=TICK_SIZE)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.yticks(fontsize=TICK_SIZE)

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------
    plt.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=LEGEND_SIZE,
        frameon=False
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    fname = f"{dataset}_{ml_model}_{attack_type}_WORST_CASE.png"
    out_path = os.path.join(output_dir, fname)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    plt.close()

###################################################################################################################################

# Summarizing ML results

def summarize_ml_results(all_files, pattern, base_path, base_ml_path):
    rows = []

    for file in all_files:
        info = utils.extract_info_from_filename(file, pattern)
        if info is None:
            continue

        file_path = os.path.join(base_path, file)
        df = pd.read_csv(file_path)
        df = unpack_tracking_info(df, tracking_col="tracking_info")

        # Global best row
        best = df.loc[df["mean_loss"].idxmin()]

        rows.append({
            "Data": info["dataset"],
            "ML_model": info["ml"],
            "round": info["seed"],
            "k": info["k"],
            "n_clusters_set": info["n_cluster"],
            "n_cluster_get": best["num_clusters"],
            "best_mean_loss": best["mean_loss"],
        })

    summary_df = pd.DataFrame(rows)

    summary_df.sort_values(
        ["Data", "ML_model", "k", "n_clusters_set", "round"],
        inplace=True
    )

    summary_df.reset_index(drop=True, inplace=True)

    output_path = os.path.join(base_ml_path, "global_best_by_round.csv")
    summary_df.to_csv(output_path, index=False)

    return summary_df

def load_baseline_results(
    csv_path,
    feature_set,           # "ALL" or "QIS"
):
    df = pd.read_csv(csv_path)
    df = df[df["Aggregate_Method"] == "Average"]

    return (
        df.assign(
            Feature_Set=feature_set,
            Process_Type="BASELINE",
            n_clusters_set=np.nan,
            n_cluster_get=np.nan,
            average_best_mean_loss=df["Loss"],
            min_best_mean_loss=df["Loss"],
        )
        .loc[:, [
            "Dataset",
            "ML_model",
            "Feature_Set",
            "Process_Type",
            "n_clusters_set",
            "n_cluster_get",
            "average_best_mean_loss",
            "min_best_mean_loss",
        ]]
    )

def load_mo_obam_results(
    csv_path,
    feature_set,     # "ALL" or "QIS"
):
    """
    Load MO-OBAM results and align schema with final_df_all
    """

    df = pd.read_csv(csv_path)

    # -------------------------
    # Keep only needed aggregates
    # -------------------------
    df = df[df["Aggregate_Method"].isin(["Average", "Minimum"])]

    # -------------------------
    # Pivot: Average & Minimum → columns
    # -------------------------
    df_pivot = (
        df
        .pivot_table(
            index=["Dataset", "ML_model", "n_clusters"],
            columns="Aggregate_Method",
            values="Loss",
            aggfunc="mean"
        )
        .reset_index()
    )

    # -------------------------
    # Rename & align schema
    # -------------------------
    df_final = (
        df_pivot
        .rename(columns={
            "n_clusters": "n_clusters_set",
            "Average": "average_best_mean_loss",
            "Minimum": "min_best_mean_loss",
        })
        .assign(
            Feature_Set=feature_set,
            Process_Type="MO-OBAM",
            n_cluster_get=np.nan
        )
        .loc[:, [
            "Dataset",
            "ML_model",
            "Feature_Set",
            "Process_Type",
            "n_clusters_set",
            "n_cluster_get",
            "average_best_mean_loss",
            "min_best_mean_loss",
        ]]
    )

    return df_final

def load_closed_loop_results(
    base_dir,
    experiment_subdir,
    dataset_names,
    ml_models,
    feature_set,          # "ALL" or "QIS"
    process_type          # "CLOSED_LOOP_WITH_REPAIR", "CLOSED_LOOP_NO_REPAIR"
):
    summary_rows = []

    for dataset in dataset_names:
        for ml_model in ml_models:
            path = os.path.join(
                base_dir,
                experiment_subdir,
                dataset,
                "ML Results",
                ml_model,
                "global_best_by_round.csv"
            )

            if not os.path.exists(path):
                print(f"⚠️ Missing: {path}")
                continue

            df = pd.read_csv(path)

            grouped = (
                df
                .groupby("n_clusters_set")
                .agg(
                    min_best_mean_loss=("best_mean_loss", "min"),
                    average_best_mean_loss=("best_mean_loss", "mean"),
                    n_cluster_get=("n_cluster_get", "mean"),
                )
                .reset_index()
            )

            for _, row in grouped.iterrows():
                summary_rows.append({
                    "Dataset": dataset,
                    "ML_model": ml_model,
                    "Feature_Set": feature_set,
                    "Process_Type": process_type,
                    "n_clusters_set": int(row["n_clusters_set"]),
                    "n_cluster_get": row["n_cluster_get"],
                    "average_best_mean_loss": row["average_best_mean_loss"],
                    "min_best_mean_loss": row["min_best_mean_loss"],
                })

    return pd.DataFrame(summary_rows)

def combine_experiments(dfs):
    combined = pd.concat(dfs, ignore_index=True)

    return combined.sort_values(
        by=[
            "Dataset",
            "ML_model",
            "Feature_Set",
            "Process_Type",
            "n_clusters_set"
        ],
        key=lambda x: x.where(x.notna(), -1)
    )

def plot_loss_comparison(
    final_df,
    feature_set,
    dataset_names,
    ml_models,
    output_path=None,
    show_plot=True
):

    # ---- Visual parameters (kept exactly as before) ----
    FIG_SIZE = (120, 80)
    LINE_WIDTH = 80
    MARKER_SIZE = 150
    LABEL_SIZE = 210
    TICK_SIZE = 210
    LEGEND_SIZE = 210

    PROCESS_STYLE = {
        "BASELINE": {
            "color": "red",
            "linestyle": "--",
            "marker": None
        },
        "CLOSED_LOOP_WITH_REPAIR": {
            "color": "blue",
            "linestyle": "-",
            "marker": "o"
        },
        "MO-OBAM": {
            "color": "green",
            "linestyle": "-",
            "marker": "s"
        },
        "CLOSED_LOOP_WITHOUT_REPAIR": {
            "color": "orange",
            "linestyle": "-",
            "marker": "^"
        }
    }

    PROCESS_LABEL = {
        "BASELINE": "Baseline",
        "CLOSED_LOOP_WITH_REPAIR": "Closed-Loop (With Repair)",
        "CLOSED_LOOP_WITHOUT_REPAIR": "Closed-Loop (No Repair)",
        "MO-OBAM": "MO-OBAM",
    }

    for dataset in dataset_names:
        for ml in ml_models:

            df_sub = final_df[
                (final_df["Dataset"] == dataset) &
                (final_df["ML_model"] == ml) &
                (final_df["Feature_Set"] == feature_set)
            ]

            if df_sub.empty:
                continue

            plt.figure(figsize=FIG_SIZE)

            x_ticks = set()
            marker_map = {}

            for process_type, df_p in df_sub.groupby("Process_Type"):

                style = PROCESS_STYLE.get(process_type.upper())

                if style is None:
                    raise ValueError(f"Unknown process type: {process_type}")

                if process_type.upper() == "BASELINE":
                    baseline_loss = df_p["average_best_mean_loss"].iloc[0]

                    plt.axhline(
                        y=baseline_loss,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=LINE_WIDTH,
                        label=PROCESS_LABEL.get(process_type.upper(), process_type)
                    )

                else:
                    df_p = df_p.sort_values("n_clusters_set")

                    x = df_p["n_clusters_set"]
                    y = df_p["average_best_mean_loss"]

                    plt.plot(
                        x,
                        y,
                        marker=style["marker"],
                        markersize=MARKER_SIZE,
                        linewidth=LINE_WIDTH,
                        linestyle=style["linestyle"],
                        color=style["color"],
                        label = PROCESS_LABEL.get(process_type.upper(), process_type)
                    )

                    x_ticks.update(x.tolist())

            plt.xlabel("Number of Clusters", fontsize=LABEL_SIZE)
            plt.ylabel("Classification Loss", fontsize=LABEL_SIZE)
            plt.tick_params(axis="both", labelsize=TICK_SIZE)

            if x_ticks:
                plt.xticks(sorted(x_ticks))

            plt.grid(True)
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=2,
                fontsize=LEGEND_SIZE,
                frameon=False
            )

            plt.tight_layout()


            if output_path:
                filename = f"{dataset}_{ml}_{feature_set}_comparison.png"
                plt.savefig(
                    os.path.join(output_path, filename),
                    bbox_inches="tight"
                )

            
            if show_plot:
                plt.show()
            else:
                plt.close()


def plot_pct_change_comparison(
    final_df,
    feature_set,
    dataset_names,
    ml_models,
    output_path=None,
    show_plot=True
):

    # ---- Visual parameters (UNCHANGED) ----
    FIG_SIZE = (160, 80)
    LABEL_SIZE = 210
    TICK_SIZE = 210
    LEGEND_SIZE = 210

    BAR_WIDTH = 0.25

    PROCESS_STYLE = {
        "CLOSED_LOOP_WITH_REPAIR": {"color": "blue"},
        "CLOSED_LOOP_WITHOUT_REPAIR": {"color": "orange"},
        "MO-OBAM": {"color": "green"},
    }

    PROCESS_LABEL = {
        "CLOSED_LOOP_WITH_REPAIR": "Closed-Loop (With Repair)",
        "CLOSED_LOOP_WITHOUT_REPAIR": "Closed-Loop (No Repair)",
        "MO-OBAM": "MO-OBAM",
    }

    # ======================================================
    # STEP 1: Precompute Y-axis limits PER DATASET
    # ======================================================
    DATASET_YLIMS = {}

    for dataset in dataset_names:
        df_d = final_df[
            (final_df["Dataset"] == dataset) &
            (final_df["Feature_Set"] == feature_set) &
            (final_df["Process_Type"] != "BASELINE") &
            (final_df['ML_model'].isin(ml_models))
        ]

        if df_d.empty:
            continue

        max_abs = df_d["pct_change_vs_baseline"].abs().max()
        DATASET_YLIMS[dataset] = max_abs  # small padding

    # ======================================================
    # STEP 2: Plot using dataset-level Y limits
    # ======================================================
    for dataset in dataset_names:
        if dataset not in DATASET_YLIMS:
            continue

        for ml in ml_models:

            df_sub = final_df[
                (final_df["Dataset"] == dataset) &
                (final_df["ML_model"] == ml) &
                (final_df["Feature_Set"] == feature_set) &
                (final_df["Process_Type"] != "BASELINE")
            ]

            if df_sub.empty:
                continue

            plt.figure(figsize=FIG_SIZE)

            # ---- X positions ----
            cluster_values = sorted(
                df_sub["n_clusters_set"]
                .dropna()
                .astype(int)
                .unique()
            )
            x = np.arange(len(cluster_values))

            # ---- Plot bars ----
            for i, (process, style) in enumerate(PROCESS_STYLE.items()):

                df_p = df_sub[df_sub["Process_Type"] == process]

                y_vals = []
                for k in cluster_values:
                    val = df_p.loc[
                        df_p["n_clusters_set"] == k,
                        "pct_change_vs_baseline"
                    ]
                    y_vals.append(val.iloc[0] if not val.empty else np.nan)

                plt.bar(
                    x + (i - 1) * BAR_WIDTH,
                    y_vals,
                    width=BAR_WIDTH,
                    color=style["color"],
                    label=PROCESS_LABEL[process]
                )

            # ---- Reference line ----
            plt.axhline(0, color="black", linewidth=5)

            # ---- Axes & labels ----
            plt.xlabel("Number of Clusters", fontsize=LABEL_SIZE)
            plt.ylabel("Change vs. Baseline (%)", fontsize=LABEL_SIZE)

            plt.xticks(x, cluster_values)
            plt.ylim(-DATASET_YLIMS[dataset], DATASET_YLIMS[dataset])

            plt.gca().yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: f"{y:.0f}")
            )

            plt.tick_params(axis="both", labelsize=TICK_SIZE)
            plt.grid(
                axis="y",
                linestyle="--",
                linewidth=4,
                alpha=0.5
            )
            plt.gca().set_axisbelow(True)

            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fontsize=LEGEND_SIZE,
                frameon=False
            )

            plt.tight_layout()

            # ---- Save / Show ----
            if output_path:
                filename = f"{dataset}_{ml}_{feature_set}_pct_change_bar.png"
                plt.savefig(
                    os.path.join(output_path, filename),
                    bbox_inches="tight"
                )

            if show_plot:
                plt.show()
            else:
                plt.close()

###################################################################################################################################

# Plotting functions and file management

def extract_info_from_filename(filename, pattern):
    match = re.match(pattern, filename)
    if match:
        return {
            "dataset": match.group("dataset"),
            "ml": match.group("ml"),
            "k": int(match.group("k")),
            "n_cluster": int(match.group("n_cluster")),
            "seed": int(match.group("seed")),
            "pid":int(match.group("pid"))
        }
    return None


##############################################################################################################

# Fixing column names
def fix_column_names(all_files, pattern, base_path):

    EXPECTED_COLS = 16
    TRACK_COLUMNS = [
        "iteration",
        "particle",
        "mean_loss",
        "num_valid_clusters",
        "num_violates_k_only",
        "num_violates_l_only",
        "num_violates_both",
        "num_records_valid",
        "num_records_violates_k_only",
        "num_records_violates_l_only",
        "num_records_violates_both",
        "total_num_violated_records_before_adjusting",
        "num_fixed_clusters",
        "num_unfixed_clusters",
        "total_num_violated_records_after_adjusting",
        "num_clusters",
    ]

    for file in all_files:
        info = extract_info_from_filename(file, pattern)

        print(
            f"Fixing: Data={info['dataset']}, "
            f"k={info['k']}, n_cluster={info['n_cluster']}, "
            f"ML={info['ml']}, Round={info['seed']}"
        )

        file_path = os.path.join(base_path, file)
        fixed_path = file_path.replace(".csv", "_fixed.csv")

        with open(file_path, "r") as f:
            lines = f.readlines()

        # ---- Safety check: column count in data ----
        max_cols = max(len(line.rstrip("\n").split(",")) for line in lines)

        if max_cols != EXPECTED_COLS:
            print(f"⚠ Warning: {file} has {max_cols} columns (expected 16)")

        # ---- Replace header ----
        lines[0] = ",".join(TRACK_COLUMNS) + "\n"

        with open(fixed_path, "w") as f:
            f.writelines(lines)

        print(f"  ✔ Saved fixed file → {os.path.basename(fixed_path)}\n")


def unpack_tracking_info(df, tracking_col="tracking_info"):
    """
    Unpack JSON-encoded tracking_info column into separate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a JSON string column.
    tracking_col : str, default="tracking_info"
        Column name that stores JSON-encoded tracking information.

    Returns
    -------
    pd.DataFrame
        DataFrame with tracking_info unpacked into separate columns.
    """

    if tracking_col not in df.columns:
        raise ValueError(f"Column '{tracking_col}' not found in dataframe.")

    # Parse JSON strings safely
    parsed = df[tracking_col].apply(
        lambda x: json.loads(x) if isinstance(x, str) else {}
    )

    # Expand JSON into columns
    tracking_df = pd.json_normalize(parsed)

    # Merge and drop original column
    df_out = pd.concat(
        [df.drop(columns=[tracking_col]), tracking_df],
        axis=1
    )

    return df_out