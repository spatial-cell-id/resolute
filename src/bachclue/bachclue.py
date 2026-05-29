import numpy as np
import scanpy as sc
import pandas as pd
import multiprocessing
from scipy import sparse
import warnings
from sklearn.metrics import calinski_harabasz_score
from scipy.signal import argrelextrema
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def _make_lightweight_adata(original_adata, use_rep, neighbors_key=None):
    """Creates a memory-efficient AnnData object containing only necessary graph/embedding data."""
    n_cells = original_adata.n_obs
    adata = sc.AnnData(X=sparse.csr_matrix((n_cells, 0)))
    
    if use_rep in original_adata.obsm:
        adata.obsm[use_rep] = original_adata.obsm[use_rep]
        
    if neighbors_key is not None and neighbors_key in original_adata.uns:
        adata.uns[neighbors_key] = original_adata.uns[neighbors_key].copy()
        conn_key = original_adata.uns[neighbors_key].get('connectivities_key', 'connectivities')
        dist_key = original_adata.uns[neighbors_key].get('distances_key', 'distances')
        
        if conn_key in original_adata.obsp:
            adata.obsp[conn_key] = original_adata.obsp[conn_key]
        if dist_key in original_adata.obsp:
            adata.obsp[dist_key] = original_adata.obsp[dist_key]
            
    return adata


def compute_clustering_score(r, light_adata, score_value, clustering_algorithm, use_rep, neighbors_key):
    """Compute clustering score for a given resolution."""
    adata = light_adata.copy() 
    clustering_name = f"{clustering_algorithm}_res{r:.2f}"

    if clustering_algorithm == 'leiden':
        sc.tl.leiden(adata, neighbors_key=neighbors_key, key_added=clustering_name, resolution=r, flavor="igraph", directed=False)
    elif clustering_algorithm == 'louvain':
        sc.tl.louvain(adata, neighbors_key=neighbors_key, key_added=clustering_name, resolution=r)
    else:
        raise ValueError("Please choose 'leiden' or 'louvain' as clustering_algorithm")

    n_clusters = adata.obs[clustering_name].nunique()
    n_points = len(adata.obs[clustering_name])
    n_dimensions = adata.obsm[use_rep].shape[1]

    if score_value == 'bic':
        n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1
        loglikelihood = sum(
            len(X_cluster) * np.log(len(X_cluster))
            - len(X_cluster) * np.log(n_points)
            - (len(X_cluster) * n_dimensions / 2) * np.log(2 * np.pi * variance)
            - (len(X_cluster) - 1) / 2
            for cluster_id in adata.obs[clustering_name].unique()
            if (X_cluster := adata.obsm[use_rep][(adata.obs[clustering_name] == cluster_id).values]).shape[0] > 1
            and (variance := np.var(X_cluster, axis=0).sum()) > 0
        )
        score_value = -2 * (loglikelihood - (n_parameters / 2) * np.log(n_points))

    elif score_value == 'calinski':
        score_value = -1 * calinski_harabasz_score(adata.obsm[use_rep], adata.obs[clustering_name])
    
    return r, score_value, n_clusters


def clustering_score(original_adata, 
                     score_value='bic', 
                     clustering_algorithm='leiden',
                     neighbors_key=None,
                     use_rep='X_pca', 
                     min_res=0.1, 
                     max_res=2.0, 
                     step=0.1,
                     n_jobs=-1,
                     plot=True):
    """Compute clustering scores over a range of resolutions in parallel using Joblib."""
    sc.settings.verbosity = 0
                  
    if use_rep not in original_adata.obsm:
        available_keys = list(original_adata.obsm.keys())
        raise KeyError(f"'{use_rep}' not found in adata.obsm. Available keys are: {available_keys}")
    
    if neighbors_key is None:
        if 'neighbors' in original_adata.uns and isinstance(original_adata.uns['neighbors'], dict) and 'connectivities_key' in original_adata.uns['neighbors']:
            neighbors_key = 'neighbors'
        else:
            rep_suffix = use_rep.replace('X_', '')
            if rep_suffix in original_adata.uns and isinstance(original_adata.uns[rep_suffix], dict) and 'connectivities_key' in original_adata.uns[rep_suffix]:
                neighbors_key = rep_suffix
            else:
                raise KeyError("A valid neighbors graph could not be auto-detected in adata.uns.")
        
    resolutions = np.arange(min_res, max_res, step)
    
    max_cores = multiprocessing.cpu_count()
    actual_jobs = max_cores if n_jobs == -1 else min(n_jobs, max_cores)
    num_cores = min(actual_jobs, len(resolutions))
    print(f"Performing {len(resolutions)} parallel clusterings, using {num_cores}/{max_cores} CPU cores")
    
    light_adata = _make_lightweight_adata(original_adata, use_rep, neighbors_key)
    
    results_gen = Parallel(n_jobs=num_cores, backend="loky", return_as="generator")(
        delayed(compute_clustering_score)(r, light_adata, score_value, clustering_algorithm, use_rep, neighbors_key) 
        for r in resolutions
    )
    results = list(tqdm(results_gen, total=len(resolutions), desc=f"Optimizing {score_value.upper()}"))
    
    df = pd.DataFrame(results, columns=['resolution', score_value, 'n_clusters'])
    raw_scores = df[score_value].values
    minima_indices = argrelextrema(raw_scores, np.less, order=1)[0]

    if len(minima_indices) > 0:
        suggested_resolutions = df.iloc[minima_indices]
        best_idx = suggested_resolutions[score_value].idxmin()
        best_res = df.loc[best_idx, 'resolution']
        print(f"Detected {len(minima_indices)} local minima at resolutions: {suggested_resolutions['resolution'].tolist()}")
        print(f"Suggested Resolution (Deepest Dip): {best_res:.2f}")
    else:
        best_res = df.iloc[df[score_value].idxmin()]['resolution']
        print(f"No local minima detected. Falling back to global minimum: {best_res:.2f}")
        
    if plot:
        ax = df.plot(x='resolution', y=score_value, linestyle='-', color='tab:blue', alpha=1.0)
        if len(minima_indices) > 0:
            plt.scatter(df.iloc[minima_indices]['resolution'], 
                        df.iloc[minima_indices][score_value], 
                        color='red', s=30, label='Local Minima', zorder=5)
        plt.axvline(x=best_res, color='red', linestyle='--', alpha=0.5, label=f'Best Resolution ({best_res})')
        plt.legend()
        plt.title(f"Clustering Score Optimization ({score_value})")
        plt.show()
    
    return df, best_res


def compute_stability_for_resolution(resolution, light_adata, clustering_algorithm='leiden', neighbors_key=None, n_iterations=50, use_rep='X_pca'):
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    adata = light_adata.copy()
    n_cells = adata.n_obs
    
    # Ground Truth Evaluation on the full pre-existing graph structure
    if clustering_algorithm == 'leiden':
        sc.tl.leiden(adata, resolution=resolution, neighbors_key=neighbors_key, key_added='original_cluster', flavor="igraph", directed=False)
    else:
        sc.tl.louvain(adata, resolution=resolution, neighbors_key=neighbors_key, key_added='original_cluster')
        
    orig_cluster_labels = adata.obs['original_cluster'].values.astype(str)
    unique_orig_clusters = np.sort(adata.obs['original_cluster'].unique().astype(int))
    
    # History matrices to track exactly what happens in each bootstrap iteration
    H_samp = np.zeros((n_cells, n_iterations), dtype=np.int8)
    H_clust = np.full((n_cells, n_iterations), -1, dtype=np.int32)
    
    # Bypass Scanpy's connectivities naming bug by falling back to None 
    # if the requested key is the default 'neighbors'.
    safe_neighbors_key = None if neighbors_key == 'neighbors' else neighbors_key
    
    for t in range(n_iterations):
        subset_indices = np.random.choice(n_cells, size=int(0.8 * n_cells), replace=False)
        
        subset_light = sc.AnnData(
            X=sparse.csr_matrix((len(subset_indices), 0)), 
            obsm={use_rep: adata.obsm[use_rep][subset_indices]}
        )
        
        sc.pp.neighbors(subset_light, use_rep=use_rep, key_added=safe_neighbors_key) 
        
        if clustering_algorithm == 'leiden':
            sc.tl.leiden(subset_light, resolution=resolution, neighbors_key=safe_neighbors_key, key_added='cluster', flavor="igraph", directed=False)
        else:
            sc.tl.louvain(subset_light, resolution=resolution, neighbors_key=safe_neighbors_key, key_added='cluster')
        
        if subset_light.obs['cluster'].dtype in ['category', 'object']:
            boot_codes = subset_light.obs['cluster'].astype('category').cat.codes.values
        else:
            boot_codes = subset_light.obs['cluster'].values.astype(int)
            
        H_samp[subset_indices, t] = 1
        H_clust[subset_indices, t] = boot_codes

    # Reconstruct exact pair-wise consensus metrics cluster by cluster on-the-fly
    cluster_stats = []
    cluster_to_indices = {cl: np.where(orig_cluster_labels == str(cl))[0] for cl in unique_orig_clusters}
    
    for i in unique_orig_clusters:
        idx_i = cluster_to_indices[i]
        if len(idx_i) <= 1:
            cluster_stats.append(np.nan)
            continue
            
        # Self-stability: block_ii math reconstructed exactly
        H_samp_i = H_samp[idx_i, :].astype(np.float32)
        block_ii_samp = H_samp_i @ H_samp_i.T
        
        block_ii_co = np.zeros((len(idx_i), len(idx_i)), dtype=np.float32)
        H_clust_i = H_clust[idx_i, :]
        for t in range(n_iterations):
            vals_t = H_clust_i[:, t]
            unique_vals = np.unique(vals_t)
            for k in unique_vals:
                if k == -1: continue
                mask = (vals_t == k)
                block_ii_co += np.outer(mask, mask)
                
        block_ii_samp[block_ii_samp == 0] = 1.0
        block_ii_consensus = block_ii_co / block_ii_samp
        np.fill_diagonal(block_ii_consensus, np.nan)
        stability_i = np.nanmean(block_ii_consensus)
        
        # Confusion blocks: block_ij math reconstructed exactly
        max_confusion_i = 0.0
        for j in unique_orig_clusters:
            if i == j: continue
            idx_j = cluster_to_indices[j]
            if len(idx_j) == 0: continue
            
            H_samp_j = H_samp[idx_j, :].astype(np.float32)
            block_ij_samp = H_samp_i @ H_samp_j.T
            
            block_ij_co = np.zeros((len(idx_i), len(idx_j)), dtype=np.float32)
            H_clust_j = H_clust[idx_j, :]
            for t in range(n_iterations):
                vals_i = H_clust_i[:, t]
                vals_j = H_clust_j[:, t]
                common_vals = np.intersect1d(vals_i, vals_j)
                for k in common_vals:
                    if k == -1: continue
                    block_ij_co += np.outer(vals_i == k, vals_j == k)
                    
            block_ij_samp[block_ij_samp == 0] = 1.0
            confusion_ij = np.mean(block_ij_co / block_ij_samp)
            if confusion_ij > max_confusion_i:
                max_confusion_i = confusion_ij
                
        score_i = stability_i - max_confusion_i
        cluster_stats.append(score_i)
        
    global_score = np.nanmean(cluster_stats)
    return resolution, global_score, len(unique_orig_clusters)


def run_parallel_stability(adata, min_res=0.1, max_res=2.0, step=0.2, use_rep=None, neighbors_key=None, clustering_algorithm='leiden', n_iterations=50, n_jobs=-1):
    sc.settings.verbosity = 0

    if use_rep is None:
        if 'X_pca' in adata.obsm:
            use_rep = 'X_pca'
        elif 'X_scVI' in adata.obsm:
            use_rep = 'X_scVI'
        else:
            pca_alts = [k for k in adata.obsm.keys() if 'X_pca' in k]
            if pca_alts:
                use_rep = pca_alts[0]
            else:
                raise ValueError("No valid latent space found for stability analysis.")
    
    # Detect original neighbors_key properly so I don't drop it
    if neighbors_key is None:
        if 'neighbors' in adata.uns and isinstance(adata.uns['neighbors'], dict) and 'connectivities_key' in adata.uns['neighbors']:
            neighbors_key = 'neighbors'
        else:
            rep_suffix = use_rep.replace('X_', '')
            if rep_suffix in adata.uns and isinstance(adata.uns[rep_suffix], dict) and 'connectivities_key' in adata.uns[rep_suffix]:
                neighbors_key = rep_suffix
            else:
                neighbors_key = 'neighbors'
    
    resolutions = np.arange(min_res, max_res, step)

    max_cores = multiprocessing.cpu_count()
    actual_jobs = max_cores if n_jobs == -1 else min(n_jobs, max_cores)
    num_cores = min(actual_jobs, len(resolutions))
    
    print(f"Performing {len(resolutions)} parallel bootstraps ({n_iterations} iterations each), using {num_cores}/{max_cores} CPU cores")
    
    light_adata = _make_lightweight_adata(adata, use_rep=use_rep, neighbors_key=neighbors_key)

    results_gen = Parallel(n_jobs=num_cores, backend='loky', return_as="generator")(
        delayed(compute_stability_for_resolution)(res, light_adata, clustering_algorithm, neighbors_key, n_iterations, use_rep)
        for res in resolutions
    )
    results = list(tqdm(results_gen, total=len(resolutions), desc="Evaluating Stability"))
    
    df_results = pd.DataFrame(results, columns=['resolution', 'penalized_score', 'n_clusters'])
    raw_scores = df_results['penalized_score'].values
    maxima_indices = argrelextrema(raw_scores, np.greater, order=1)[0]
    
    if len(maxima_indices) > 0:
        suggested_resolutions = df_results.iloc[maxima_indices]
        best_idx = suggested_resolutions['penalized_score'].idxmax()
        best_res = df_results.loc[best_idx, 'resolution']
        print(f"Detected {len(maxima_indices)} stable local maxima at resolutions: {suggested_resolutions['resolution'].tolist()}")
        print(f"Suggested Stability Resolution (Highest Peak): {best_res:.2f}")
    else:
        best_res = df_results.loc[df_results['penalized_score'].idxmax(), 'resolution']
        print(f"No local maxima detected. Falling back to global maximum: {best_res:.2f}")
        
    return df_results, best_res


def plot_bachclue(df_score, best_res_score, score_metric, df_stab=None, best_res_stab=None):
    num_plots = 2 if df_stab is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]
        
    def _draw_axis(ax, df, metric_col, best_res, title):
        ax.plot(df['resolution'], df[metric_col], marker='o', linestyle='-', color='tab:blue', label=metric_col)
        ax.set_xlabel('Resolution', fontsize=12)
        ax.set_ylabel(f'Score ({metric_col})', color='tab:blue', fontsize=12)
        ax.tick_params(axis='y', labelcolor='tab:blue')
        scores = df[metric_col].values

        if metric_col in ['bic', 'calinski', 'BIC score', 'Calinski-Harabasz score']:
            extrema_indices = argrelextrema(scores, np.less, order=1)[0]
            extrema_label, extrema_color = 'Local Minima', 'red'
        else:
            extrema_indices = argrelextrema(scores, np.greater, order=1)[0]
            extrema_label, extrema_color = 'Local Maxima', 'green'

        if len(extrema_indices) > 0:
            ax.scatter(df.iloc[extrema_indices]['resolution'], 
                       df.iloc[extrema_indices][metric_col], 
                       color=extrema_color, s=60, label=extrema_label, zorder=5)

        ax.axvline(x=best_res, color='black', linestyle='--', alpha=0.8, label=f'Best Res: {best_res:.2f}')
        ax.legend(loc='best')
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

    _draw_axis(axes[0], df_score, score_metric, best_res_score, f"{score_metric.upper()} Score Optimization")
    if df_stab is not None:
        _draw_axis(axes[1], df_stab, 'penalized_score', best_res_stab, "Bootstrap Stability Optimization")
    plt.tight_layout()
    plt.show()

    
def run_bachclue(adata, 
                 score_value='bic', 
                 clustering_algorithm='leiden',
                 use_rep='X_pca',
                 stability_use_rep=None,
                 neighbors_key=None,
                 min_res=0.1, 
                 max_res=2.0, 
                 step=0.1,
                 compute_stability=True,
                 n_iterations=50,
                 n_jobs=-1,
                 plot=True):
    """Main execution wrapper for BaCHClue clustering optimization."""

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    print(f"--- Running {score_value.upper()} Scoring ---")
    df_score, best_res_score = clustering_score(
        original_adata=adata,
        score_value=score_value,
        clustering_algorithm=clustering_algorithm,
        neighbors_key=neighbors_key,  
        use_rep=use_rep,        
        min_res=min_res,
        max_res=max_res,
        step=step,
        n_jobs=n_jobs,
        plot=False  
    )
    
    df_stab, best_res_stab = None, None
    if compute_stability:
        print("\n--- Running Bootstrap Stability Analysis ---")
        df_stab, best_res_stab = run_parallel_stability(
            adata=adata,
            min_res=min_res,
            max_res=max_res,
            step=step,
            use_rep=stability_use_rep,  
            clustering_algorithm=clustering_algorithm,
            neighbors_key=neighbors_key,              
            n_iterations=n_iterations,
            n_jobs=n_jobs
        )

    if plot:
        plot_bachclue(df_score, best_res_score, score_value, df_stab, best_res_stab)
        
    if df_stab is not None:
        df_merged = pd.merge(df_score, df_stab, on='resolution', suffixes=('_clust_score', '_stab'))
        print(f"\nOptimization Summary for {score_value.upper()} & Stability")
        print("-" * 101)
        header = f"{'Resolution':<10} | {f'{score_value.upper()} Score':<16} | {'Clusters (clustering_score)':<27} | {'Stability Score':<16} | {'Clusters (Stability)':<20}"
        print(header)
        print("-" * len(header))
        for _, row in df_merged.iterrows():
            print(
                f"{row['resolution']:<10.1f} | "
                f"{row[score_value]:<16.4f} | "
                f"{int(row['n_clusters_clust_score']):<27} | "
                f"{row['penalized_score']:<16.4f} | "
                f"{int(row['n_clusters_stab']):<20}"
            )
        print("-" * len(header))
    else:
        print(f"\nOptimization Summary for {score_value.upper()}")
        print("-" * 59)
        header = f"{'Resolution':<10} | {f'{score_value.upper()} Score':<16} | {'Clusters (clustering_score)':<27}"
        print(header)
        print("-" * len(header))
        for _, row in df_score.iterrows():
            print(
                f"{row['resolution']:<10.1f} | "
                f"{row[score_value]:<16.4f} | "
                f"{int(row['n_clusters']):<27}"
            )
        print("-" * len(header))
        
    print("\n")
    return {
        'score_results': df_score,
        'best_score_res': best_res_score,
        'stability_results': df_stab,
        'best_stability_res': best_res_stab
    }