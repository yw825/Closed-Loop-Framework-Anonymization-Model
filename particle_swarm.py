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
from collections import defaultdict
from sklearn.base import clone
import copy
from scipy.stats import entropy as scipy_entropy
import gower
import random

from constants import *
import utils
import model_train
import concurrent.futures

# Function to compute numerical distance
def get_numeric_distance(df, NQIs, particle):
    # Convert the dataframe columns to a NumPy array for faster operations
    df_values = df[NQIs].values  # Shape: (num_rows, num_NQIs)
    
    # Shape of particle: (num_centroids, num_NQIs)
    # Broadcast the centroid values across all rows in df to calculate distances
    diffs = df_values[:, np.newaxis, :] - particle[:, :len(NQIs)]  # Shape: (num_rows, num_centroids, num_NQIs)
    squared_diffs = diffs ** 2  # Squared differences
    num_dist = np.sum(squared_diffs, axis=2)  # Sum over the NQIs (axis=2)

    return num_dist

# Function to compute categorical distance
def get_categorical_distance(df, CQIs, NQIs, particle):
    # Create a matrix of size (len(df), num_particles) for categorical distance
    categorical_dist = np.zeros((len(df), particle.shape[0]))

    # Extract categorical data from df
    categorical_data = df[CQIs].values

    # Extract the centroids for categorical data (for each particle)
    centroids = particle[:, len(NQIs):]  # Get the categorical columns from particle

    # Compare categorical values using broadcasting: (categorical_data != centroids) returns a matrix of True/False
    diffs = (categorical_data[:, None, :] != centroids[None, :, :]).astype(int)

    # Sum the differences for each row to get the categorical distance for each particle
    categorical_dist = np.sum(diffs, axis=2)

    return categorical_dist

# Function to compute the total distance
def get_total_distance(df, CQIs, NQIs, particle, gamma):
    numeric_distance = get_numeric_distance(df, NQIs, particle)
    categorical_distance = get_categorical_distance(df, CQIs, NQIs, particle)

    # Convert the distances into DataFrames for alignment by PatientIdentifier
    numeric_df = pd.DataFrame(numeric_distance, index=df.index)
    categorical_df = pd.DataFrame(categorical_distance, index=df.index)

    total_distance = numeric_df + gamma * categorical_df
    return total_distance

# Function to get the minimum distance and cluster assignment
def get_min_distance(df, CQIs, NQIs, particle, gamma):
    total_distance = get_total_distance(df, CQIs, NQIs, particle, gamma)
    # min_distance = np.min(total_distance, axis=1)
    cluster_assignment = np.argmin(total_distance, axis=1)
    return cluster_assignment # min_distance

def satisfies_k_anonymity(cluster_df, k_val):
    return len(cluster_df) >= k_val

def entropy(series):
    probs = series.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-12))  # add epsilon to avoid log(0)

def satisfies_l_diversity(cluster_df, SAs, l, check_each_sa=True, check_composite=True, composite_strict=False):
    log_l = np.log2(l)
    # log_l = np.log2(1.5) # Lower the value l (1-2) if SAs are binary
    satisfies_l = True
    
    if check_each_sa:
        sa_entropies = cluster_df[SAs].apply(entropy)
        if not (sa_entropies >= log_l).all():
            satisfies_l = False
            # return satisfies_l

    if check_composite:
        composite_sa = cluster_df[SAs].astype(str).agg('|'.join, axis=1)
        if composite_strict:
            if entropy(composite_sa) < log_l:
                satisfies_l = False
                # return satisfies_l
        else:
            if composite_sa.nunique() < l:
                satisfies_l = False
                # return satisfies_l
    return satisfies_l


def classify_clusters(df, k_val, SAs, l):
    clusters = df.groupby('cluster')

    violates_both = []
    violates_k_only = []
    violates_l_only = []
    valid_clusters = []

    for cluster_index, cluster_data in clusters:
        satisfies_k = satisfies_k_anonymity(cluster_data, k_val)
        satisfies_l = satisfies_l_diversity(cluster_data, SAs, l)

        if not satisfies_k and not satisfies_l:
            violates_both.append(cluster_data)
        elif not satisfies_k:
            violates_k_only.append(cluster_data)
        elif not satisfies_l:
            violates_l_only.append(cluster_data)
        else:
            valid_clusters.append(cluster_data)

    return violates_both, violates_k_only, violates_l_only, valid_clusters

def split_valid_clusters(valid_clusters, k_val, SAs, l):
    retained_records = []
    excess_pool = []

    for cluster_data in valid_clusters:
        cluster_data = cluster_data.copy()
        found_valid_subset = False

        # Try all possible sizes from k to len(cluster)
        for size in range(k_val, len(cluster_data) + 1):
            subset = cluster_data.sample(n=size, random_state=42)
            satisfies_l = satisfies_l_diversity(subset, SAs, l)

            if satisfies_k_anonymity(subset, k_val) and satisfies_l:
                retained_records.append(subset)
                remaining = cluster_data.drop(subset.index)
                excess_pool.append(remaining)
                found_valid_subset = True
                break

        if not found_valid_subset:
            # If we couldn't find a valid subset (shouldn't happen if originally valid), retain all
            retained_records.append(cluster_data)

    # retained_df = pd.concat(retained_records, ignore_index=True)
    # excess_df = pd.concat(excess_pool, ignore_index=True) if excess_pool else pd.DataFrame()

    return retained_records, excess_pool

def get_centroid_values_from_particle(particle, cluster_list):
    centroid_info = []
    for cluster in cluster_list:
        if len(cluster) == 0:
            continue
        cluster_index = cluster['cluster'].values[0]
        centroid_vector = particle[cluster_index, :]
        centroid_info.append((centroid_vector, cluster_index))
    return centroid_info  # List of (vector, index)


def find_closest_centroid_to_pool(particle, excess_pool, violates_both, violates_k_only, violates_l_only):
    
    # Get pool centroids and their cluster indices
    pool_info = get_centroid_values_from_particle(particle, excess_pool)
    pool_centroids = [vec for vec, idx in pool_info]
    pool_indices = [idx for vec, idx in pool_info]


    # Get violated centroids and their cluster indices
    violated_centroids = []
    violated_indices = []
    for violated in [violates_both, violates_k_only, violates_l_only]:
        if len(violated) != 0:
            info = get_centroid_values_from_particle(particle, violated)
            violated_centroids.extend([vec for vec, idx in info])
            violated_indices.extend([idx for vec, idx in info])

    # Convert to DataFrames for Gower distance
    pool_df = pd.DataFrame(pool_centroids)
    violated_df = pd.DataFrame(violated_centroids)

    # Compute Gower distance matrix
    pool_distances = gower.gower_matrix(violated_df, pool_df)  # shape: (len(violated), len(pool))

    # Find closest pool centroid for each violated cluster
    # Create a grid of all combinations
    violated_grid, pool_grid = np.meshgrid(violated_indices, pool_indices, indexing='ij')

    # Flatten the arrays
    violated_flat = violated_grid.ravel()
    pool_flat = pool_grid.ravel()
    dist_flat = pool_distances.ravel()

    # Construct DataFrame
    distance_df = pd.DataFrame({
        'violated_cluster': violated_flat,
        'pool_cluster': pool_flat,
        'distance': dist_flat
    })

    # Sort by distance ascending
    sorted_distances = distance_df.sort_values(by=['violated_cluster', 'distance'], ascending=[True, True]).reset_index(drop=True)

    return sorted_distances

def fix_violated_clusters(violates_both, violates_k_only, violates_l_only, valid_clusters, SAs, k_val, particle, l):
    sorted_distances = find_closest_centroid_to_pool(particle, valid_clusters, violates_both, violates_k_only, violates_l_only)

    # Step 1: Convert violated clusters into a dictionary using 'cluster' column as the key
    violated_clusters_dict = {
        int(df['cluster'].iloc[0]): df.copy() for df in violates_both + violates_k_only + violates_l_only
        if not df.empty and 'cluster' in df.columns
    }
    # print(f"📦 Total violated clusters to fix: {len(violated_clusters_dict)}")

    # Step 2: Convert excess_pool clusters into a dictionary using 'cluster' column as the key
    excess_pool_dict = {
        int(df['cluster'].iloc[0]): df.copy()
        for df in valid_clusters
        if not df.empty and 'cluster' in df.columns
    }
    # print(f"🎯 Initial excess pool clusters: {len(excess_pool_dict)}")

    # Dictionary to store fixed violated clusters
    fixed_clusters = {}
    unfixed_clusters = {}

    # Step 3: Iterate over each unique violated cluster ID
    for violated_id in sorted_distances['violated_cluster'].unique():
        # print(f"\n🔧 Fixing violated cluster: {violated_id}")
        violated_df = violated_clusters_dict[violated_id]

        # Step 4: Get sorted pool candidates for this violated cluster
        candidates = sorted_distances[sorted_distances['violated_cluster'] == violated_id]
        # print(f"➡️  Candidate pool clusters for violated {violated_id}: {candidates['pool_cluster'].tolist()}")

        fixed = False  # Track if the current violated cluster gets fixed

        # Step 5: Try to fix using nearest pool clusters
        for _, row in candidates.iterrows():
            pool_id = row['pool_cluster']
            # print(f"   🔍 Trying pool cluster: {pool_id}")

            if pool_id not in excess_pool_dict:
                # print(f"   ❌ Pool cluster {pool_id} not found in excess_pool_dict.")
                continue

            pool_df = excess_pool_dict[pool_id]
            if pool_df.empty:
                # print(f"   ⚠️ Pool cluster {pool_id} is empty.")
                continue

            # Step 6: Add records from this pool to violated cluster until it satisfies constraints
            move_count = 0
            satisfies_l = satisfies_l_diversity(violated_df, SAs, l)
            while not (satisfies_k_anonymity(violated_df, k_val) and satisfies_l):
                if pool_df.empty:
                    # print(f"   ⚠️ Pool cluster {pool_id} ran out of records while fixing cluster {violated_id}.")
                    break

                record_to_move = pool_df.sample(n=1, random_state=42)
                pool_df = pool_df.drop(record_to_move.index).reset_index(drop=True)
                violated_df = pd.concat([violated_df, record_to_move], ignore_index=True)
                move_count += 1

                satisfies_l = satisfies_l_diversity(violated_df, SAs, l)

            # print(f"   ✅ Moved {move_count} record(s) from pool {pool_id} to violated {violated_id}")

            # Step 7: Update dicts
            excess_pool_dict[pool_id] = pool_df
            violated_df['cluster'] = violated_id
            violated_clusters_dict[violated_id] = violated_df

            # Step 8: Check if fix is successful
            if satisfies_k_anonymity(violated_df, k_val) and satisfies_l_diversity(violated_df, SAs, l):
                # print(f"🎉 Successfully fixed cluster {violated_id} using pool {pool_id}")
                fixed_clusters[violated_id] = violated_df
                fixed = True
                break
            # else:
            #     print(f"❌ Cluster {violated_id} still not valid after trying pool {pool_id}")

        if not fixed:
            # print(f"⚠️ Could not fix violated cluster {violated_id} with any available pool clusters.")
            unfixed_clusters[violated_id] = violated_df

    updated_excess_pool = [df for df in excess_pool_dict.values() if not df.empty]
    # print(f"\n🔄 Remaining clusters in excess pool after fixing: {len(updated_excess_pool)}")
    # print(f"🚨 Unfixed violated clusters count: {len(unfixed_clusters)}")

    return fixed_clusters, updated_excess_pool, unfixed_clusters

def verify_valid_clusters(valid_clusters, updated_excess_pool, unfixed_clusters, k_val, SAs, l):
    # Step 1: Create a dictionary for valid clusters
    valid_dict = {}
    for df in valid_clusters:
        cluster_id = int(df['cluster'].iloc[0])
        if cluster_id in valid_dict:
            valid_dict[cluster_id] = pd.concat([valid_dict[cluster_id], df], ignore_index=True)
        else:
            valid_dict[cluster_id] = df.copy()
    # print(f"🔍 Valid clusters count: {len(valid_dict)}")

    # Step 2: Merge excess pool records back into respective clusters
    for df in updated_excess_pool:
        cluster_id = int(df['cluster'].iloc[0])
        if cluster_id in valid_dict:
            valid_dict[cluster_id] = pd.concat([valid_dict[cluster_id], df], ignore_index=True)
        else:
            valid_dict[cluster_id] = df.copy()

    # Step 3: Re-check validity of all clusters
    verified_valid_clusters = []
    updated_unfixed_clusters = unfixed_clusters.copy()  # avoid mutating original dict

    for cluster_id, df in valid_dict.items():
        satisfies_k = satisfies_k_anonymity(df, k_val)
        satisfies_l = satisfies_l_diversity(df, SAs, l)

        if satisfies_k and satisfies_l:
            verified_valid_clusters.append(df)
        else:
            updated_unfixed_clusters[cluster_id] = df
            # print(f"⚠️ Cluster {cluster_id} no longer satisfies k/l. Added to unfixed_clusters.")

    return verified_valid_clusters, updated_unfixed_clusters


def apply_centroids(df, particle, CQIs, NQIs):
    anonymized_data = []
    for cluster_index in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_index].copy()
        cluster_data[NQIs] = cluster_data[NQIs].astype(float)  # Ensure NQIs are float for calculations
        centroid_values = particle[cluster_index]
        n_rows = len(cluster_data)
        cluster_data.loc[:, NQIs] = np.tile(centroid_values[:len(NQIs)], (n_rows, 1))
        cluster_data.loc[:, CQIs] = np.tile(centroid_values[len(NQIs):], (n_rows, 1))
        anonymized_data.append(cluster_data)
    return pd.concat(anonymized_data)


def get_adaptive_anonymized_data(df, CQIs, NQIs, particle, gamma, k_val, SAs, l):
    updated_unfixed_clusters_list = []
    tracking_info = {}

    # Assign clusters using min distance
    cluster_assignment = get_min_distance(df, CQIs, NQIs, particle, gamma)
    df = df.copy()
    df['cluster'] = cluster_assignment
    # print("------------------------------------------------------")
    # print("Original data:")
    # print(f"Number of CLUSTERS in the original data: ",len(np.unique(cluster_assignment)))

    # Split into valid and violated clusters
    violates_both, violates_k_only, violates_l_only, valid_clusters = classify_clusters(df, k_val, SAs, l)
    tracking_info["num_valid_clusters"] = len(valid_clusters)
    tracking_info["num_violates_k_only"] = len(violates_k_only)
    tracking_info["num_violates_l_only"] = len(violates_l_only)
    tracking_info["num_violates_both"] = len(violates_both)
    # print(f"Number of valid CLUSTERS: ", tracking_info["num_valid_clusters"])
    # print(f"Number of violated k only CLUSTERS: ", tracking_info["num_violates_k_only"])
    # print(f"Number of violated l only CLUSTERS: ", tracking_info["num_violates_l_only"])
    # print(f"Number of violated both CLUSTERS: ", tracking_info["num_violates_both"])

    tracking_info["num_records_valid"] = sum(len(cluster) for cluster in valid_clusters)
    tracking_info["num_records_violates_k_only"] = sum(len(cluster) for cluster in violates_k_only)
    tracking_info["num_records_violates_l_only"] = sum(len(cluster) for cluster in violates_l_only)
    tracking_info["num_records_violates_both"] = sum(len(cluster) for cluster in violates_both)
    tracking_info["total_num_violated_records_before_adjusting"] = tracking_info["num_records_violates_k_only"] + tracking_info["num_records_violates_l_only"] + tracking_info["num_records_violates_both"]
    # print(f"Number of records in valid clusters: ", tracking_info["num_records_valid"])
    # print(f"Number of records in violated k only clusters: ", tracking_info["num_records_violates_k_only"])
    # print(f"Number of records in violated l only clusters: ", tracking_info["num_records_violates_l_only"])
    # print(f"Number of records in violated both clusters: ", tracking_info["num_records_violates_both"])
    # print(f"Number of total records in violated clusters: ", tracking_info["total_num_violated_records_before_adjusting"])

    if tracking_info["num_records_valid"] == len(df):
        # No need to fix anything
        anonymized_df = apply_centroids(df, particle, CQIs, NQIs)
        still_violated = pd.DataFrame()
        tracking_info["total_num_violated_records_after_adjusting"] = len(still_violated)
        tracking_info["num_clusters"] = len(np.unique(anonymized_df['cluster']))
    else:
        # Retain k and l from each valid cluster
        retained, excess_pool = split_valid_clusters(valid_clusters, k_val, SAs, l)
        # tracking_info["num_retained_clusters"] = len(retained)
        # tracking_info["num_excess_clusters"] = len(excess_pool)
        # print("------------------------------------------------------")
        # print("Splitting valid clusters starts here:")
        # print(f"num_retained_CLUSTERS: ", len(retained))
        # print(f"num_excess_CLUSTERS: ", len(excess_pool))

        # Fix violated clusters 
        fixed, remaining_pool, still_violated = fix_violated_clusters(violates_both, violates_k_only, violates_l_only, excess_pool, SAs, k_val, particle, l)
        # tracking_info["num_fixed_clusters"] = len(fixed)
        # tracking_info["num_unfixed_clusters"] = len(still_violated)
        # print("------------------------------------------------------")
        # print("Fixing violated clusters starts here:")
        # print("num_fixed_CLUSTERS: ", len(fixed))
        # print("num_CLUSTERS_left_in_the_pool: ", len(remaining_pool))
        # print("num_CLUSTERS_still_violated: ", len(still_violated))

        # Verify valid clusters are still valid after adjusting
        verified_valid_clusters, updated_unfixed_clusters = verify_valid_clusters(retained, remaining_pool, still_violated, k_val, SAs, l)
        tracking_info["num_fixed_clusters"] = len(verified_valid_clusters) 
        tracking_info["num_unfixed_clusters"] = len(updated_unfixed_clusters)
        tracking_info["total_num_violated_records_after_adjusting"] = sum(len(cluster) for cluster in updated_unfixed_clusters.values())
        # print("------------------------------------------------------")
        # print("Verifying if valid clusters are still valid:")
        # print(f"num_fixed_CLUSTERS: ", tracking_info["num_fixed_clusters"])
        # print(f"num_unfixed_CLUSTERS: ", tracking_info["num_unfixed_clusters"])
        # print(f"total_num_violated_records_after_adjusting", tracking_info["total_num_violated_records_after_adjusting"])


        # Combine everything before anonymizing
        fixed_clusters_list = list(fixed.values())
        updated_unfixed_clusters_list = list(updated_unfixed_clusters.values()) 
        final_df = pd.concat(fixed_clusters_list+verified_valid_clusters+updated_unfixed_clusters_list, ignore_index=True)
        # Anonymize the data
        anonymized_df = apply_centroids(final_df, particle, CQIs, NQIs)
        tracking_info["num_clusters"] = len(np.unique(anonymized_df['cluster']))
        # print("------------------------------------------------------")
        # print("Anonymization finished here:")
        # print("Number of CLUSTERS in anonymized data: ", tracking_info["num_clusters"])
        # print(anonymized_df.shape)

    return anonymized_df, tracking_info, updated_unfixed_clusters_list


def initialize_particles(n_population, NQIs, CQIs, bounds, df, n_cluster):

    particles = np.empty((n_population, n_cluster, len(NQIs) + len(CQIs)), dtype=object)

    # Generate random values for NQIs (numerical)
    for i, nqi in enumerate(NQIs):
        lower_bound = bounds[nqi]['lower_bound']
        upper_bound = bounds[nqi]['upper_bound']

        # Randomly generate values within bounds for each cluster (2 clusters)
        particles[:, :, i] = np.random.randint(lower_bound, upper_bound, size=(n_population, n_cluster))
        # particles[:, :, i] = np.random.uniform(lower_bound, upper_bound, size=(n_population, n_cluster))
        
    # Generate random values for CQIs (categorical)
    for i, cqi in enumerate(CQIs):
        unique_values = df[cqi].dropna().unique()

        # Randomly assign values for each cluster from the unique categorical values
        particles[:, :, len(NQIs) + i] = np.random.choice(unique_values, size=(n_population, n_cluster))

    return particles

def update_categorical_variables(particle_categorical, CQIs, centv, levels):

    # Ensure centv is a 2D array (n_particles, n_categories)
    centv = np.array(centv, dtype=float)
    
    # Saremi, S., Mirjalili, S., & Lewis, A. (2015). 
    # How important is a transfer function in discrete heuristic algorithms. 
    # Neural Computing and Applications, 26, 625-640.
    # Calculate the T value for each element
    T = np.abs(centv / np.sqrt(centv**2 + 1))

    # Generate random values for each particle
    rand = np.random.uniform(0, 1, size=particle_categorical.shape)

    # Compare rand with T for each element, determining whether to update the category
    mask = rand < T

    for i, cqi in enumerate(CQIs):
        random_choice = np.random.choice(list(levels[cqi].keys()), size=particle_categorical.shape[0:2])
        particle_categorical[:,:, i] = np.where(mask[:,:, i], random_choice, particle_categorical[:,:, i])

    return particle_categorical


def check_bound(particle_numeric, lower_bounds, upper_bounds, column_means):
    # # Ensure particle_numeric is a float type to perform comparisons
    # particle_numeric = np.array(particle_numeric, dtype=float)

    # Apply masks for out-of-bound values for each column
    for col_idx in range(particle_numeric.shape[2]):  # Iterate over columns
        mask_low = particle_numeric[:,:, col_idx] < lower_bounds[col_idx]
        mask_high = particle_numeric[:,:, col_idx] > upper_bounds[col_idx]

        # Replace out-of-bound values with the corresponding column mean
        particle_numeric[mask_low, col_idx] = column_means[col_idx]
        particle_numeric[mask_high, col_idx] = column_means[col_idx]

    return particle_numeric.astype(float)  # Convert back to integer values if needed


def update_particles_velocity_and_location(particles, n_population, centv, pbest, global_best, NQIs, CQIs, levels, bounds, nqi_means):
    uc = np.random.uniform(0, 0.01, size=(n_population, 1, 1))
    ud = np.random.uniform(0, 0.01, size=(n_population, 1, 1))
    c = 1 - uc - ud 

    centv = np.array(centv, dtype=float)
    centv[:,:,:len(NQIs)] = c * np.array(centv)[:,:,:len(NQIs)] + uc * (np.array(pbest)[:,:,:len(NQIs)] - np.array(particles)[:,:,:len(NQIs)]) + \
                        ud * (np.array(global_best)[:,:len(NQIs)] - np.array(particles)[:,:,:len(NQIs)])

    # Update numeric variables in particles based on the velocities
    particles = np.array(particles)
    particles[:,:,:len(NQIs)] = np.array(particles)[:,:,:len(NQIs)] + centv[:,:,:len(NQIs)]

    # Ensure particles stay within bounds
    lower_bounds = np.array([bounds[NQI]['lower_bound'] for NQI in NQIs])
    upper_bounds = np.array([bounds[NQI]['upper_bound'] for NQI in NQIs])
    # Apply check_bound function to all particles
    particles[:,:,:len(NQIs)] = check_bound(particles[:,:,:len(NQIs)], lower_bounds, upper_bounds, nqi_means)

    ########################################################################################################
    # Update categorical velocities

    l = len(NQIs)
    r = l + len(CQIs)
    global_best = np.array(global_best)
    pbest = np.array(pbest)
    centv[:,:, l:r] = c * centv[:,:, l:r] + uc * (np.where(pbest[:,:, l:r] == particles[:,:, l:r], 0, 1)) + \
                        ud * (np.where(global_best[:,l:r] == particles[:,:, l:r], 0, 1))       

    # Update categorical variables in particles
    particles[:,:, l:r] = update_categorical_variables(particles[:,:,l:r], CQIs, centv[:,:,l:r], levels)
    
    return particles, centv

##################################################################################################################################

# Adaptive PSO with particle reduction

def improvement_based_stop(history, patience=50, epsilon=1e-3):
    if len(history) < patience + 1:
        return False

    recent = history[-(patience + 1):]
    improvements = np.abs(np.diff(recent))

    return np.all(improvements < epsilon)

def diversity_based_stop(
    particles,
    initial_diversity,
    n_numeric,
    ratio_threshold=0.01
):
    """
    particles: shape (n_population, n_cluster, n_numeric + n_categorical)
    Uses ONLY numeric QIs to compute diversity
    """

    # 1️⃣ Extract numeric QIs only
    numeric_particles = particles[:, :, :n_numeric].astype(float)
    # shape: (n_population, n_cluster, n_numeric)

    # 2️⃣ Flatten cluster structure per particle
    numeric_particles = numeric_particles.reshape(
        numeric_particles.shape[0], -1
    )
    # shape: (n_population, n_cluster * n_numeric)

    # 3️⃣ Compute swarm centroid in numeric space
    centroid = np.mean(numeric_particles, axis=0)

    # 4️⃣ Distance of each particle to centroid
    distances = np.linalg.norm(
        numeric_particles - centroid,
        axis=1
    )

    # 5️⃣ Diversity = average distance
    diversity = np.mean(distances)

    # 6️⃣ Relative stopping condition
    return (diversity / initial_diversity) < ratio_threshold

def compute_initial_diversity(particles, n_numeric):
    numeric_particles = particles[:, :, :n_numeric].astype(float)
    numeric_particles = numeric_particles.reshape(
        numeric_particles.shape[0], -1
    )

    centroid = np.mean(numeric_particles, axis=0)
    distances = np.linalg.norm(numeric_particles - centroid, axis=1)

    return np.mean(distances)

def reduce_particles(
    X, V, pbest, pbest_val,
    keep_ratio=0.7
):
    """
    Keep top-performing particles
    """
    n_keep = max(5, int(len(X) * keep_ratio))
    elite_idx = np.argsort(pbest_val)[:n_keep]

    return (
        X[elite_idx],
        V[elite_idx],
        pbest[elite_idx],
        pbest_val[elite_idx],
    )


##################################################################################################################################

# Closed-Loop without repairing process

def get_anonymized_data_without_repairing(
    df,
    CQIs,
    NQIs,
    particle,
    gamma,
    k_val,
    SAs,
    l
):
    tracking_info = {}

    # -------------------------
    # Assign clusters
    # -------------------------
    cluster_assignment = get_min_distance(df, CQIs, NQIs, particle, gamma)
    df = df.copy()
    df["cluster"] = cluster_assignment

    # -------------------------
    # Classify clusters (NO fixing)
    # -------------------------
    violates_both, violates_k_only, violates_l_only, valid_clusters = classify_clusters(
        df, k_val, SAs, l
    )

    # -------------------------
    # Cluster-level tracking
    # -------------------------
    tracking_info["num_valid_clusters"] = len(valid_clusters)
    tracking_info["num_violates_k_only"] = len(violates_k_only)
    tracking_info["num_violates_l_only"] = len(violates_l_only)
    tracking_info["num_violates_both"] = len(violates_both)

    # -------------------------
    # Record-level tracking
    # -------------------------
    tracking_info["num_records_valid"] = sum(len(c) for c in valid_clusters)
    tracking_info["num_records_violates_k_only"] = sum(len(c) for c in violates_k_only)
    tracking_info["num_records_violates_l_only"] = sum(len(c) for c in violates_l_only)
    tracking_info["num_records_violates_both"] = sum(len(c) for c in violates_both)

    tracking_info["total_num_violated_records"] = (
        tracking_info["num_records_violates_k_only"]
        + tracking_info["num_records_violates_l_only"]
        + tracking_info["num_records_violates_both"]
    )

    tracking_info["num_clusters"] = len(np.unique(cluster_assignment))

    # -------------------------
    # Anonymize by centroids ONLY
    # -------------------------
    anonymized_df = apply_centroids(df, particle, CQIs, NQIs)


    return anonymized_df, tracking_info


def run_particle_swarm_experiment_without_repairing(
    df,
    dataset_name,
    ml_name,
    ml_model,
    params,
    dataset_config,
    base_path,
    current_iter
):
    # import time
    # start_time = time.time()

    # =========================
    # Unpack parameters
    # =========================
    # PSO
    n_population = params["n_population"]
    maxIter = params["maxIter"]
    n_bootstrap = params["n_bootstrap"]

    warmup_ratio = params["warmup_ratio"]
    adaptive_ratio = params["adaptive_ratio"]

    keep_ratio = params["keep_ratio"]
    elite_ratio = params["elite_ratio"]

    patience_phase2 = params["patience_phase2"]
    epsilon_phase2 = params["epsilon_phase2"]
    ratio_threshold = params["ratio_threshold"]

    patience_phase3 = params["patience_phase3"]
    epsilon_phase3 = params["epsilon_phase3"]

    # time_budget = params["time_budget"]

    n_cluster = params["n_cluster"]
    l_val = dataset_config["l"]

    # Anonymization
    gamma = params["gamma"]
    k_val = params["k"]
    # initial_violation_threshold = params["initial_violation_threshold"]
    # violation_decay_rate = params["violation_decay_rate"]
    # penalty_weight = params["penalty_weight"]
    aggregate_function = params["aggregate_function"]

    # Dataset
    SAs = dataset_config["SAs"]
    NQIs = dataset_config["NQIs"]
    CQIs = dataset_config["CQIs"]

    bounds = utils.get_nqi_bounds(df, NQIs)
    levels = utils.get_cqi_levels(df, CQIs)
    nqi_means = df[NQIs].mean().values  

    # =========================
    # Initialization
    # =========================
    centv = np.zeros((n_population, n_cluster, len(NQIs) + len(CQIs)), dtype=object)
    fit = np.zeros(n_population)

    global_best_fit = float("inf")
    pbest_fit = np.full(n_population, np.inf)
    pbest = np.zeros((n_population, n_cluster, len(NQIs) + len(CQIs)), dtype=object)

    particles = initialize_particles(n_population, NQIs, CQIs, bounds, df, n_cluster)

    history = []

    warmup_iters = int(warmup_ratio * maxIter)
    adaptive_iters = int(adaptive_ratio * maxIter)
    elite_size = max(5, int(elite_ratio * n_population))

    initial_diversity = None

    # =========================
    # Tracking file
    # =========================
    track_folder = os.path.join(
        base_path,
        ml_name,
        "Tracking Info"
    )
    os.makedirs(track_folder, exist_ok=True)

    pid = os.getpid()
    track_file = os.path.join(
        track_folder,
        f"track_{dataset_name}_{ml_name}_k{k_val}_ncluster{n_cluster}"
        f"_round{current_iter}_pid{pid}.csv"
    )
    if os.path.exists(track_file):
        os.remove(track_file)

    first_write = True

    validated_track_file = False

    # =========================
    # Main loop
    # =========================
    for iteration in range(maxIter):

        # # ---- Time budget ----
        # if time.time() - start_time > time_budget:
        #     print(f"⏱ Time budget reached at iteration {iteration}")
        #     break

        # ---- Phase ----
        if iteration < warmup_iters:
            phase = "warmup"
        elif iteration < adaptive_iters:
            phase = "adaptive"
        else:
            phase = "exploitation"

        # violation_threshold = max(
        #     initial_violation_threshold - iteration * violation_decay_rate, 0
        # )

        # ---- Evaluate particles ----
        for i in range(n_population):

            anonymized_df, tracking_info = (
                get_anonymized_data_without_repairing(
                    df, CQIs, NQIs, particles[i], gamma, k_val, SAs, l_val
                )
            )

            anonymized_df_encoded = utils.encode_categorical_from_file(anonymized_df)

            losses = model_train.train_model_bootstrap(
                anonymized_df_encoded,
                dataset_name,
                clone(ml_model),
                n_bootstrap
            )

            # excess_violation = max(0, len(violating_records) - violation_threshold)
            # penalty = penalty_weight * excess_violation

            fit[i] = (
                np.mean(losses) if aggregate_function == "mean" else np.max(losses)
            )  # + penalty

            if fit[i] < pbest_fit[i]:
                pbest_fit[i] = fit[i]
                pbest[i] = particles[i]

            # ---- Stream tracking info ----
            row = {
                "iteration": iteration,
                "particle": i,
                "mean_loss": np.mean(losses),
                "tracking_info": json.dumps(tracking_info)
            }

            pd.DataFrame([row]).to_csv(
                track_file,
                mode="a",
                header=first_write,
                index=False
            )

            first_write = False

            # 🔒 Validate file format ONCE (after first write)
            if not validated_track_file:
                df_check = pd.read_csv(track_file)

                expected_cols = ["iteration", "particle", "mean_loss", "tracking_info"]

                if list(df_check.columns) != expected_cols:
                    raise RuntimeError(
                        f"Corrupted tracking file: {track_file}\n"
                        f"Got columns: {list(df_check.columns)}"
                    )

                validated_track_file = True

        # ---- Global best ----
        if global_best_fit > np.min(fit):
            global_best_fit = np.min(fit)
            global_best = particles[np.argmin(fit)]

        history.append(global_best_fit)

        # ---- Initial diversity ----
        if phase == "warmup" and iteration == warmup_iters - 1:
            initial_diversity = compute_initial_diversity(
                particles, n_numeric=len(NQIs)
            )

        # ---- Phase 2: particle reduction ----
        if phase == "adaptive" and iteration % 3 == 0 and n_population > elite_size:

            stagnating = improvement_based_stop(
                history, patience_phase2, epsilon_phase2
            )

            redundant = diversity_based_stop(
                particles, initial_diversity,
                n_numeric=len(NQIs),
                ratio_threshold=ratio_threshold
            )

            if stagnating or redundant:
                particles, centv, pbest, pbest_fit = reduce_particles(
                    X=particles,
                    V=centv,
                    pbest=pbest,
                    pbest_val=pbest_fit,
                    keep_ratio=keep_ratio
                )
                n_population = particles.shape[0]
                fit = fit[:n_population]

        # ---- Phase 3: early stopping ----
        if phase == "exploitation":
            if improvement_based_stop(
                history, patience_phase3, epsilon_phase3
            ):
                print(f"Early stop in exploitation at iteration {iteration}")
                break

        # ---- Update swarm ----
        particles, centv = update_particles_velocity_and_location(
            particles, n_population, centv, pbest, global_best,
            NQIs, CQIs, levels, bounds, nqi_means
        )

        if iteration % 3 == 0:
            gc.collect()

    # =========================
    # Save best result
    # =========================
    best_df = get_anonymized_data_without_repairing(
        df, CQIs, NQIs, global_best, gamma, k_val, SAs, l_val
    )[0]

    out_dir = os.path.join(base_path, ml_name)
    os.makedirs(out_dir, exist_ok=True)

    filename = f"best_{dataset_name}_{ml_name}_k{k_val}_ncluster{n_cluster}_round{current_iter}.csv"
    best_df.to_csv(os.path.join(out_dir, filename), index=False)

    gc.collect()


##################################################################################################################################

# Closed-Loop with repairing process

def run_particle_swarm_experiment_with_repairing(
    df,
    dataset_name,
    ml_name,
    ml_model,
    params,
    dataset_config,
    base_path,
    current_iter
):
    # import time
    # start_time = time.time()

    # =========================
    # Unpack parameters
    # =========================
    # PSO
    n_population = params["n_population"]
    maxIter = params["maxIter"]
    n_bootstrap = params["n_bootstrap"]

    warmup_ratio = params["warmup_ratio"]
    adaptive_ratio = params["adaptive_ratio"]

    keep_ratio = params["keep_ratio"]
    elite_ratio = params["elite_ratio"]

    patience_phase2 = params["patience_phase2"]
    epsilon_phase2 = params["epsilon_phase2"]
    ratio_threshold = params["ratio_threshold"]

    patience_phase3 = params["patience_phase3"]
    epsilon_phase3 = params["epsilon_phase3"]

    # time_budget = params["time_budget"]

    n_cluster = params["n_cluster"]
    l_val = dataset_config["l"]

    # Anonymization
    gamma = params["gamma"]
    k_val = params["k"]
    initial_violation_threshold = params["initial_violation_threshold"]
    violation_decay_rate = params["violation_decay_rate"]
    penalty_weight = params["penalty_weight"]
    aggregate_function = params["aggregate_function"]

    # Dataset
    SAs = dataset_config["SAs"]
    NQIs = dataset_config["NQIs"]
    CQIs = dataset_config["CQIs"]

    bounds = utils.get_nqi_bounds(df, NQIs)
    levels = utils.get_cqi_levels(df, CQIs)
    nqi_means = df[NQIs].mean().values  

    # =========================
    # Initialization
    # =========================
    centv = np.zeros((n_population, n_cluster, len(NQIs) + len(CQIs)), dtype=object)
    fit = np.zeros(n_population)

    global_best_fit = float("inf")
    pbest_fit = np.full(n_population, np.inf)
    pbest = np.zeros((n_population, n_cluster, len(NQIs) + len(CQIs)), dtype=object)

    particles = initialize_particles(n_population, NQIs, CQIs, bounds, df, n_cluster)

    history = []

    warmup_iters = int(warmup_ratio * maxIter)
    adaptive_iters = int(adaptive_ratio * maxIter)
    elite_size = max(5, int(elite_ratio * n_population))

    initial_diversity = None

    # =========================
    # Tracking file
    # =========================
    track_folder = os.path.join(
        base_path,
        ml_name,
        "Tracking Info"
    )
    os.makedirs(track_folder, exist_ok=True)

    pid = os.getpid()
    track_file = os.path.join(
        track_folder,
        f"track_{dataset_name}_{ml_name}_k{k_val}_ncluster{n_cluster}"
        f"_round{current_iter}_pid{pid}.csv"
    )
    if os.path.exists(track_file):
        os.remove(track_file)

    first_write = True

    validated_track_file = False

    # =========================
    # Main loop
    # =========================
    for iteration in range(maxIter):

        # # ---- Time budget ----
        # if time.time() - start_time > time_budget:
        #     print(f"⏱ Time budget reached at iteration {iteration}")
        #     break

        # ---- Phase ----
        if iteration < warmup_iters:
            phase = "warmup"
        elif iteration < adaptive_iters:
            phase = "adaptive"
        else:
            phase = "exploitation"

        violation_threshold = max(
            initial_violation_threshold - iteration * violation_decay_rate, 0
        )

        # ---- Evaluate particles ----
        for i in range(n_population):

            anonymized_df, tracking_info, violating_records = (
                get_adaptive_anonymized_data(
                    df, CQIs, NQIs, particles[i], gamma, k_val, SAs, l_val
                )
            )

            anonymized_df_encoded = utils.encode_categorical_from_file(anonymized_df)

            losses = model_train.train_model_bootstrap(
                anonymized_df_encoded,
                dataset_name,
                clone(ml_model),
                n_bootstrap
            )

            excess_violation = max(0, len(violating_records) - violation_threshold)
            penalty = penalty_weight * excess_violation

            fit[i] = (
                np.mean(losses) if aggregate_function == "mean" else np.max(losses)
            ) + penalty

            if fit[i] < pbest_fit[i]:
                pbest_fit[i] = fit[i]
                pbest[i] = particles[i]

            # ---- Stream tracking info ----
            row = {
                "iteration": iteration,
                "particle": i,
                "mean_loss": np.mean(losses),
                "tracking_info": json.dumps(tracking_info)
            }

            pd.DataFrame([row]).to_csv(
                track_file,
                mode="a",
                header=first_write,
                index=False
            )

            first_write = False

            # 🔒 Validate file format ONCE (after first write)
            if not validated_track_file:
                df_check = pd.read_csv(track_file)

                expected_cols = ["iteration", "particle", "mean_loss", "tracking_info"]

                if list(df_check.columns) != expected_cols:
                    raise RuntimeError(
                        f"Corrupted tracking file: {track_file}\n"
                        f"Got columns: {list(df_check.columns)}"
                    )

                validated_track_file = True

        # ---- Global best ----
        if global_best_fit > np.min(fit):
            global_best_fit = np.min(fit)
            global_best = particles[np.argmin(fit)]

        history.append(global_best_fit)

        # ---- Initial diversity ----
        if phase == "warmup" and iteration == warmup_iters - 1:
            initial_diversity = compute_initial_diversity(
                particles, n_numeric=len(NQIs)
            )

        # ---- Phase 2: particle reduction ----
        if phase == "adaptive" and iteration % 3 == 0 and n_population > elite_size:

            stagnating = improvement_based_stop(
                history, patience_phase2, epsilon_phase2
            )

            redundant = diversity_based_stop(
                particles, initial_diversity,
                n_numeric=len(NQIs),
                ratio_threshold=ratio_threshold
            )

            if stagnating or redundant:
                particles, centv, pbest, pbest_fit = reduce_particles(
                    X=particles,
                    V=centv,
                    pbest=pbest,
                    pbest_val=pbest_fit,
                    keep_ratio=keep_ratio
                )
                n_population = particles.shape[0]
                fit = fit[:n_population]

        # ---- Phase 3: early stopping ----
        if phase == "exploitation":
            if improvement_based_stop(
                history, patience_phase3, epsilon_phase3
            ):
                print(f"Early stop in exploitation at iteration {iteration}")
                break

        # ---- Update swarm ----
        particles, centv = update_particles_velocity_and_location(
            particles, n_population, centv, pbest, global_best,
            NQIs, CQIs, levels, bounds, nqi_means
        )

        if iteration % 3 == 0:
            gc.collect()

    # =========================
    # Save best result
    # =========================
    best_df = get_adaptive_anonymized_data(
        df, CQIs, NQIs, global_best, gamma, k_val, SAs, l_val
    )[0]

    out_dir = os.path.join(base_path, ml_name)
    os.makedirs(out_dir, exist_ok=True)

    filename = f"best_{dataset_name}_{ml_name}_k{k_val}_ncluster{n_cluster}_round{current_iter}.csv"
    best_df.to_csv(os.path.join(out_dir, filename), index=False)

    gc.collect()


##################################################################################################################################

# Closed-Loop with QIs only

def run_particle_swarm_experiment_QIs(
    df,
    dataset_name,
    ml_name,
    ml_model,
    params,
    dataset_config,
    base_path,
    current_iter
):
    # import time
    # start_time = time.time()

    # =========================
    # Unpack parameters
    # =========================
    # PSO
    n_population = params["n_population"]
    maxIter = params["maxIter"]
    n_bootstrap = params["n_bootstrap"]

    warmup_ratio = params["warmup_ratio"]
    adaptive_ratio = params["adaptive_ratio"]

    keep_ratio = params["keep_ratio"]
    elite_ratio = params["elite_ratio"]

    patience_phase2 = params["patience_phase2"]
    epsilon_phase2 = params["epsilon_phase2"]
    ratio_threshold = params["ratio_threshold"]

    patience_phase3 = params["patience_phase3"]
    epsilon_phase3 = params["epsilon_phase3"]

    # time_budget = params["time_budget"]

    n_cluster = params["n_cluster"]
    l_val = dataset_config["l"]

    # Anonymization
    gamma = params["gamma"]
    k_val = params["k"]
    initial_violation_threshold = params["initial_violation_threshold"]
    violation_decay_rate = params["violation_decay_rate"]
    penalty_weight = params["penalty_weight"]
    aggregate_function = params["aggregate_function"]

    # Dataset
    SAs = dataset_config["SAs"]
    NQIs = dataset_config["NQIs"]
    CQIs = dataset_config["CQIs"]

    bounds = utils.get_nqi_bounds(df, NQIs)
    levels = utils.get_cqi_levels(df, CQIs)
    nqi_means = df[NQIs].mean().values  

    # =========================
    # Initialization
    # =========================
    centv = np.zeros((n_population, n_cluster, len(NQIs) + len(CQIs)), dtype=object)
    fit = np.zeros(n_population)

    global_best_fit = float("inf")
    pbest_fit = np.full(n_population, np.inf)
    pbest = np.zeros((n_population, n_cluster, len(NQIs) + len(CQIs)), dtype=object)

    particles = initialize_particles(n_population, NQIs, CQIs, bounds, df, n_cluster)

    history = []

    warmup_iters = int(warmup_ratio * maxIter)
    adaptive_iters = int(adaptive_ratio * maxIter)
    elite_size = max(5, int(elite_ratio * n_population))

    initial_diversity = None

    # =========================
    # Tracking file
    # =========================
    track_folder = os.path.join(
        base_path,
        ml_name,
        "Tracking Info"
    )
    os.makedirs(track_folder, exist_ok=True)

    pid = os.getpid()
    track_file = os.path.join(
        track_folder,
        f"track_{dataset_name}_{ml_name}_k{k_val}_ncluster{n_cluster}"
        f"_round{current_iter}_pid{pid}.csv"
    )
    if os.path.exists(track_file):
        os.remove(track_file)

    first_write = True

    validated_track_file = False

    # =========================
    # Main loop
    # =========================
    for iteration in range(maxIter):

        # # ---- Time budget ----
        # if time.time() - start_time > time_budget:
        #     print(f"⏱ Time budget reached at iteration {iteration}")
        #     break

        # ---- Phase ----
        if iteration < warmup_iters:
            phase = "warmup"
        elif iteration < adaptive_iters:
            phase = "adaptive"
        else:
            phase = "exploitation"

        violation_threshold = max(
            initial_violation_threshold - iteration * violation_decay_rate, 0
        )

        # ---- Evaluate particles ----
        for i in range(n_population):

            anonymized_df, tracking_info, violating_records = (
                get_adaptive_anonymized_data(
                    df, CQIs, NQIs, particles[i], gamma, k_val, SAs, l_val
                )
            )

            # anonymized_df_encoded = utils.encode_categorical_from_file(anonymized_df)

            losses = model_train.train_model_bootstrap_QIs_closed_loop(
                anonymized_df,
                NQIs,
                CQIs,
                dataset_name,
                ml_model,
                n_bootstrap
            )

            excess_violation = max(0, len(violating_records) - violation_threshold)
            penalty = penalty_weight * excess_violation

            fit[i] = (
                np.mean(losses) if aggregate_function == "mean" else np.max(losses)
            ) + penalty

            if fit[i] < pbest_fit[i]:
                pbest_fit[i] = fit[i]
                pbest[i] = particles[i]

            # ---- Stream tracking info ----
            row = {
                "iteration": iteration,
                "particle": i,
                "mean_loss": np.mean(losses),
                "tracking_info": json.dumps(tracking_info)
            }

            pd.DataFrame([row]).to_csv(
                track_file,
                mode="a",
                header=first_write,
                index=False
            )

            first_write = False

            # 🔒 Validate file format ONCE (after first write)
            if not validated_track_file:
                df_check = pd.read_csv(track_file)

                expected_cols = ["iteration", "particle", "mean_loss", "tracking_info"]

                if list(df_check.columns) != expected_cols:
                    raise RuntimeError(
                        f"Corrupted tracking file: {track_file}\n"
                        f"Got columns: {list(df_check.columns)}"
                    )

                validated_track_file = True

        # ---- Global best ----
        if global_best_fit > np.min(fit):
            global_best_fit = np.min(fit)
            global_best = particles[np.argmin(fit)]

        history.append(global_best_fit)

        # ---- Initial diversity ----
        if phase == "warmup" and iteration == warmup_iters - 1:
            initial_diversity = compute_initial_diversity(
                particles, n_numeric=len(NQIs)
            )

        # ---- Phase 2: particle reduction ----
        if phase == "adaptive" and iteration % 3 == 0 and n_population > elite_size:

            stagnating = improvement_based_stop(
                history, patience_phase2, epsilon_phase2
            )

            redundant = diversity_based_stop(
                particles, initial_diversity,
                n_numeric=len(NQIs),
                ratio_threshold=ratio_threshold
            )

            if stagnating or redundant:
                particles, centv, pbest, pbest_fit = reduce_particles(
                    X=particles,
                    V=centv,
                    pbest=pbest,
                    pbest_val=pbest_fit,
                    keep_ratio=keep_ratio
                )
                n_population = particles.shape[0]
                fit = fit[:n_population]

        # ---- Phase 3: early stopping ----
        if phase == "exploitation":
            if improvement_based_stop(
                history, patience_phase3, epsilon_phase3
            ):
                print(f"Early stop in exploitation at iteration {iteration}")
                break

        # ---- Update swarm ----
        particles, centv = update_particles_velocity_and_location(
            particles, n_population, centv, pbest, global_best,
            NQIs, CQIs, levels, bounds, nqi_means
        )

        if iteration % 3 == 0:
            gc.collect()

    # =========================
    # Save best result
    # =========================
    best_df = get_adaptive_anonymized_data(
        df, CQIs, NQIs, global_best, gamma, k_val, SAs, l_val
    )[0]

    out_dir = os.path.join(base_path, ml_name)
    os.makedirs(out_dir, exist_ok=True)

    filename = f"best_{dataset_name}_{ml_name}_k{k_val}_ncluster{n_cluster}_round{current_iter}.csv"
    best_df.to_csv(os.path.join(out_dir, filename), index=False)

    gc.collect()


##################################################################################################################################

# Run a single experiment
def run_single_experiment(
    seed,
    df,
    dataset_name,
    ml_name,
    ml_model,
    params,
    dataset_config,
    base_path,
    experiment_runner
):
    np.random.seed(seed)
    random.seed(seed)

    experiment_runner(
        df=df,
        dataset_name=dataset_name,
        ml_name=ml_name,
        ml_model=ml_model,
        params=params,
        dataset_config=dataset_config,
        base_path=base_path,
        current_iter=seed
    )
    