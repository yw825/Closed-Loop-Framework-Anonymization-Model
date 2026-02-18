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
    "Data_name": {
        "path": "path you save the dataset.csv",
        "NQIs": ["nqi1"], # Numeric Quasi-Identifiers
        "CQIs": ["cqi1"], # Categorical Quasi-Identifiers
        "SAs": ["sa1"], # Sensitive Attributes
        "n_cluster": [2], # number of clusters that the anonymized data will have
        "l": 2 # If the sensitive attributes have more that 2 distinct values, set l to 2, otherwise set l to 1 < l < 2
    }
}

DATASET_NAMES = list(DATASET_CONFIGS.keys())

EXPERIMENT_SOURCES = {
# This is where you want to save the results of the experiments, you can change the path to your own path
    "CLOSED_LOOP": {
        "type": "folder",
        "base_path": "Your path to save the results of the experiments/closed_loop/"
    }
}