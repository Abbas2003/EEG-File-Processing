import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def prepare_cluster_data(df: pd.DataFrame):
    """Scale features and perform PCA."""
    # Drop metadata
    meta_cols = ['epoch_id', 'time_start', 'time_end']
    x = df.drop(columns=[c for c in meta_cols if c in df.columns])
    
    # Scale
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # PCA
    pca = PCA(n_components=0.95)
    x_pca = pca.fit_transform(x_scaled)
    
    return x_scaled, x_pca, pca, scaler

def get_optimal_k(x_scaled, k_range=(2, 8)):
    """Compute metrics to help find optimal K."""
    inertias = []
    silhouettes = []
    ks = list(range(k_range[0], k_range[1] + 1))
    
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(x_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(x_scaled, labels))
        
    return ks, inertias, silhouettes

def run_clustering(x_scaled, n_clusters=3):
    """Run K-Means and return labels."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = km.fit_predict(x_scaled)
    return labels, km

def interpret_clusters(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """
    Assign cognitive state labels using a competitive ranking approach.
    Each cluster competes for the label that best describes its relative profile.
    """
    df_temp = df.copy()
    df_temp['cluster'] = labels
    profiles = df_temp.groupby('cluster').mean()
    n_clusters = len(profiles)
    
    # 1. Define ranking metrics for each potential state
    # We want to find which cluster is 'most' of something compared to others
    rankings = {
        'Focused':   profiles['engagement_index'],
        'Relaxed':   profiles['global_alpha_rel'],
        'Distracted': profiles['global_theta_rel'],
        'Anxious':    -profiles['faa_af_alpha'] # Lower FAA often means withdrawal/anxiety
    }
    
    # 2. Assign labels using a GREEDY UNIQUE strategy
    # Ensure each label is used only once for the most distinct clusters
    cluster_map = {}
    remaining_clusters = list(range(n_clusters))
    available_labels = list(rankings.keys())
    
    # Pre-calculate all relative scores
    all_scores = {} # (cluster_idx, label) -> score
    for i in range(n_clusters):
        for state, series in rankings.items():
            mean_val = series.mean()
            std_val  = series.std() or 1e-10
            all_scores[(i, state)] = (series.iloc[i] - mean_val) / std_val

    # Greedy assignment
    while remaining_clusters and available_labels:
        # Find the best remaining (cluster, label) pair
        best_score = -999
        best_pair = None
        
        for c_idx in remaining_clusters:
            for lbl in available_labels:
                score = all_scores[(c_idx, lbl)]
                if score > best_score:
                    best_score = score
                    best_pair = (c_idx, lbl)
        
        if best_pair:
            c_idx, lbl = best_pair
            cluster_map[c_idx] = lbl
            remaining_clusters.remove(c_idx)
            available_labels.remove(lbl)
            
    # If more clusters than labels, assign remaining to 'Other' or repeat with prefix
    for c_idx in remaining_clusters:
        cluster_map[c_idx] = f"Mixed State {c_idx}"
        
    return cluster_map

def get_2d_projection(x_scaled):
    """t-SNE for visualization."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(x_scaled)-1))
    return tsne.fit_transform(x_scaled)

def plot_clusters_2d(x_2d, labels, cluster_names=None):
    """Create a scatter plot of clusters."""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    
    unique_labels = np.unique(labels)
    for l in unique_labels:
        mask = labels == l
        name = cluster_names[l] if cluster_names else f"Cluster {l}"
        ax.scatter(x_2d[mask, 0], x_2d[mask, 1], label=name, alpha=0.7, s=50)
        
    ax.legend(facecolor='#0d1117', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.set_title("Cognitive State Clusters (t-SNE)", color='#c9d1d9', fontweight='bold')
    ax.tick_params(colors='#8b949e')
    plt.tight_layout()
    return fig

def plot_timeline(labels, times, cluster_names=None):
    """Gantt-like chart of state transitions."""
    fig, ax = plt.subplots(figsize=(12, 2), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    
    # Calculate width based on average step between epochs
    width = 2.0
    if len(times) > 1:
        width = times[1] - times[0]
    
    for i in range(len(labels)):
        name = cluster_names[labels[i]] if cluster_names else str(labels[i])
        color = plt.cm.tab10(labels[i])
        ax.barh(0, width, left=times[i], color=color, edgecolor='none')
        
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", color='#8b949e')
    ax.set_title("Cognitive State Timeline", color='#c9d1d9', fontweight='bold')
    ax.tick_params(colors='#8b949e')
    plt.tight_layout()
    return fig
