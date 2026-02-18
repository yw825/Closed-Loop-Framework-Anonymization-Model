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

import utils 
import model_train
from constants import *
import particle_swarm



DATASET_CONFIGS = {
    "German_credit": {
        "path": "/Users/yusiwei/Library/CloudStorage/OneDrive-Personal/research/Third Year Paper/experiments/German Credit dataset.csv",
        "NQIs": ["age"],
        "CQIs": ["personal_status", "job"],
        "SAs": ["checking_status", "savings_status"],
        "n_cluster": [10, 15, 20, 25, 30, 35, 40, 45, 50], # 10, 15, 20, 25, 30, 35, 40, 45, 50
        "l": 2
    },

    "Adult": {
        "path": "/Users/yusiwei/Library/CloudStorage/OneDrive-Personal/research/Third Year Paper/experiments/adult.csv",
        "NQIs": ["age"],
        "CQIs": ["race", "sex", "marital_status"],
        "SAs": ["occupation"],
        "n_cluster": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        "l": 2
    },

    "Sepsis": {
        "path": "/Users/yusiwei/Library/CloudStorage/OneDrive-Personal/research/Third Year Paper/experiments/PSM-SepsisPatient.csv",
        "NQIs": ["AgeCategory", "LOSDays", "NumberofVisits"],
        "CQIs": ["GenderDescription", "RaceDescription", "EthnicGroupDescription"],
        "SAs": [
            "HX_AIDS", "HX_ALCOHOL", "HX_ANEMDEF", "HX_ARTH", "HX_BLDLOSS",
            "HX_CHF", "HX_CAD", "HX_CHRNLUNG", "HX_COAG", "HX_DEPRESS",
            "HX_DM", "HX_DMCX", "HX_DRUG", "HX_HTN", "HX_HYPOTHY",
            "HX_LIVER", "HX_LYMPH", "HX_LYTES", "HX_METS", "HX_NEURO",
            "HX_OBESE", "HX_PARA", "HX_PERIVASC", "HX_PSYCH", "HX_PULMCIRC",
            "HX_RENLFAIL", "HX_TUMOR", "HX_ULCER", "HX_VALVE", "HX_WGHTLOSS"
        ],
        "n_cluster": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        "l": 1.5
    }
}

DATASET_NAMES = list(DATASET_CONFIGS.keys())

EXPERIMENT_SOURCES = {
    "BASELINE": {
        "type": "csv",
        "path_key": "path"  # from DATASET_CONFIGS
    },
    "MO-OBAM": {
        "type": "folder",
        "base_path": "/Users/yusiwei/Library/CloudStorage/OneDrive-Personal/research/Fourth Year Paper/Experiments/4th Experiments/2 MO-OBAM Experiments"
    },
    "CLOSED_LOOP_WITH_REPAIR": {
        "type": "folder",
        "base_path": "/Users/yusiwei/Library/CloudStorage/OneDrive-Personal/research/Fourth Year Paper/Experiments/4th Experiments/3 Closed-Loop Experiments/With Repairing Process"
    },
    "CLOSED_LOOP_WITHOUT_REPAIR": {
        "type": "folder",
        "base_path": "/Users/yusiwei/Library/CloudStorage/OneDrive-Personal/research/Fourth Year Paper/Experiments/4th Experiments/3 Closed-Loop Experiments/Without Repairing Process"
    }
}