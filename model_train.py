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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, shapiro
from sklearn.model_selection import GridSearchCV
import gc
import itertools
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, log_loss, hinge_loss
)
from sklearn.dummy import DummyClassifier
from sklearn.base import clone

from constants import *

# Baseline model training with bootstrap

def get_dataset_config(df, dataset_name):
    if dataset_name == "Adult":
        target = "income_ >50K"
        drop_cols = [target] if df.shape[1] == 97 else [target, "cluster"]

    elif dataset_name == "German_credit":
        target = "credit_risk_good"
        drop_cols = [target] if df.shape[1] == 49 else [target, "cluster"]

    elif dataset_name == "Sepsis":
        target = "SepsisFlag"
        drop_cols = (
            [target, "PatientIdentifier"]
            if df.shape[1] == 117
            else [target, "PatientIdentifier", "cluster"]
        )

    return target, drop_cols

def compute_loss(model, X_test, y_test):
    """
    Returns scalar loss value.
    """

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        return log_loss(y_test, y_score)

    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
        return hinge_loss(y_test, y_score)

    else:
        raise ValueError("Model must support predict_proba or decision_function")
    
def train_model_bootstrap(
    df, dataset_name, model, n_bootstrap, test_size=0.2
):
    target, drop_cols = get_dataset_config(df, dataset_name)
    X = df.drop(columns=drop_cols)
    y = df[target]

    losses = []

    for i in range(n_bootstrap):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i
        )

        model.fit(X_train, y_train)
        loss = compute_loss(model, X_test, y_test)
        losses.append(loss)

    return losses

#############################################################################################################


def _bootstrap_loss_QIs(
    X,
    y,
    ML_model,
    n_bootstrap,
    test_size
):
    losses = []

    for i in range(n_bootstrap):
        model = clone(ML_model)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=i,
            stratify=y
        )

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            loss = log_loss(y_test, y_score)
        else:
            y_score = model.decision_function(X_test)
            loss = hinge_loss(y_test, y_score)

        losses.append(loss)

    return losses

def train_model_bootstrap_QIs(
    df,
    NQIs,
    CQIs,
    dataset_name,
    ML_model,
    n_bootstrap,
    test_size=0.2
):
    # Target
    target_map = {
        "Adult": ("income", {"<=50K": 0, ">50K": 1}),
        "German_credit": ("credit_risk", {"bad": 0, "good": 1}),
        "Sepsis": ("SepsisFlag", None),
    }

    target, mapping = target_map[dataset_name]

    X_raw = df[NQIs + CQIs]
    X = pd.get_dummies(X_raw, columns=CQIs, drop_first=True)

    y_raw = df[target].astype(str).str.strip()
    y = y_raw.map(mapping) if mapping else df[target]

    return _bootstrap_loss_QIs(
        X, y, ML_model, n_bootstrap, test_size
    )

def train_model_bootstrap_QIs_closed_loop(
    df,
    NQIs,
    CQIs,
    dataset_name,
    ML_model,
    n_bootstrap,
    test_size=0.2
):
    # EXACT same logic
    return train_model_bootstrap_QIs(
        df,
        NQIs,
        CQIs,
        dataset_name,
        ML_model,
        n_bootstrap,
        test_size
    )