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
import matplotlib.pyplot as plt


def compute_clustering_score(r, adata, score_value, clustering_algorithm, use_rep, neighbors_key):
    """Compute clustering score for a given resolution."""
    adata = adata.copy()
    clustering_name = f"{clustering_algorithm}_res{r:.2f}"

    # Perform clustering
    if clustering_algorithm == 'leiden':
        sc.tl.leiden(adata, neighbors_key=neighbors_key, key_added=clustering_name, resolution=r, flavor="igraph", directed=False)
    elif clustering_algorithm == 'louvain':
        sc.tl.louvain(adata, neighbors_key=neighbors_key, key_added=clustering_name, resolution=r)
    else:
        raise ValueError("Please choose 'leiden' or 'louvain' as clustering_algorithm")

    # Compute score
    n_clusters = adata.obs[clustering_name].nunique()
    n_points = len(adata.obs[clustering_name])
    n_dimensions = adata.obsm[use_rep].shape[1]

    if score_value == 'bic':
        score_name = "BIC score"
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
        score_name = "Calinski-Harabasz score"
        score_value = -1 * calinski_harabasz_score(adata.obsm[use_rep], adata.obs[clustering_name])
    
    return r, score_value, n_clusters


def clustering_score(original_adata, 
                     score_value='bic', 
                     clustering_algorithm='leiden',
                     neighbors_key = None, # defaults to .obsp[connectivities]
                     use_rep='X_pca', 
                     min_res=0.1, 
                     max_res=2.0, 
                     step=0.1,
                     n_jobs=-1,
                     plot=True):
    """Compute clustering scores over a range of resolutions in parallel using Joblib."""
    sc.settings.verbosity = 0  # Suppress Scanpy verbosity
                  
    
    # Validate dimensionality reduction flexibly instead of allowing only X_pca or X_umap
    # Now this supports tools like scVI which uses the latent embedding X_scVI
    if use_rep not in original_adata.obsm:
        available_keys = list(original_adata.obsm.keys())
        raise KeyError(f"'{use_rep}' not found in adata.obsm. Available keys are: {available_keys}")
    
    if neighbors_key is None:
            # 1. Prioritize the default Scanpy 'neighbors' slot if it is valid
            if 'neighbors' in original_adata.uns and isinstance(original_adata.uns['neighbors'], dict) and 'connectivities_key' in original_adata.uns['neighbors']:
                neighbors_key = 'neighbors'
            else:
                # 2. Fallback to checking a slot named after the representation suffix (e.g., 'scVI')
                rep_suffix = use_rep.replace('X_', '')
                if rep_suffix in original_adata.uns and isinstance(original_adata.uns[rep_suffix], dict) and 'connectivities_key' in original_adata.uns[rep_suffix]:
                    neighbors_key = rep_suffix
                else:
                    raise KeyError(
                        f"A valid neighbors graph could not be auto-detected in adata.uns. "
                        f"Please run sc.pp.neighbors(adata, use_rep='{use_rep}') before optimizing, "
                        f"or pass your neighbors_key explicitly."
                    )
        
    # Generate resolution range
    resolutions = np.arange(min_res, max_res, step)
    
    # Determine number of available CPU cores
    max_cores = multiprocessing.cpu_count()
    actual_jobs = max_cores if n_jobs == -1 else min(n_jobs, max_cores)
    num_cores = min(actual_jobs, len(resolutions))
    print(f"Performing {len(resolutions)} parallel clusterings, using {num_cores}/{multiprocessing.cpu_count()} CPU cores")
    
    # Stream results as a generator to allow real-time tqdm tracking
    results_gen = Parallel(n_jobs=num_cores, backend="loky", return_as="generator")(
        delayed(compute_clustering_score)(r, original_adata, score_value, clustering_algorithm, use_rep, neighbors_key) 
        for r in resolutions
    )
    results = list(tqdm(results_gen, total=len(resolutions), desc=f"Optimizing {score_value.upper()}"))
    
    df = pd.DataFrame(results, columns=['resolution', score_value, 'n_clusters'])
    raw_scores = df[score_value].values

    # Identify indices of all local minima in the raw data
    # order = 1 compares each point to its immediate left and right neighbor. If score(res-1) < score(res) < score(res+1),
    # then it's a local minimum.

    minima_indices = argrelextrema(raw_scores, np.less, order=1)[0]

    if len(minima_indices) > 0:
        # Get all resolutions that are local minima
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

def compute_stability_for_resolution(resolution, original_adata, clustering_algorithm='leiden', neighbors_key=None, n_iterations=50, use_rep='X_pca'):
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    adata = original_adata.copy()
    cells = adata.obs_names.values
    n_cells = len(cells)
    
    co_occurrence_sum = sparse.csr_matrix((n_cells, n_cells), dtype=np.float32)
    sampling_count = sparse.csr_matrix((n_cells, n_cells), dtype=np.float32)
    
    for _ in range(n_iterations):
        subset_indices = np.random.choice(n_cells, size=int(0.8 * n_cells), replace=False)
        subset = adata[subset_indices].copy()
        
        # Generates subset neighbor graph into the correct key
        sc.pp.neighbors(subset, use_rep=use_rep, key_added=neighbors_key) 
        
        # Apply the chosen clustering algorithm
        if clustering_algorithm == 'leiden':
            sc.tl.leiden(subset, resolution=resolution, neighbors_key=neighbors_key, key_added='cluster', flavor="igraph", directed=False)
        else:
            sc.tl.louvain(subset, resolution=resolution, neighbors_key=neighbors_key, key_added='cluster')
        
        if subset.obs['cluster'].dtype == 'category' or subset.obs['cluster'].dtype == 'object':
            clusters = subset.obs['cluster'].astype('category').cat.codes.values
        else:
            clusters = subset.obs['cluster'].values.astype(int)
            
        n_clusters = clusters.max() + 1
        
        data = np.ones(len(subset_indices), dtype=np.float32)
        Z = sparse.csr_matrix((data, (subset_indices, clusters)), shape=(n_cells, n_clusters))
        co_occurrence_sum += (Z @ Z.T)
        
        V = sparse.csr_matrix((np.ones(len(subset_indices), dtype=np.float32), 
                               (subset_indices, np.zeros(len(subset_indices)))), 
                              shape=(n_cells, 1))
        sampling_count += (V @ V.T)

    # Ground Truth Evaluation on full copy using the correct algorithm and neighbors_key
    if clustering_algorithm == 'leiden':
        sc.tl.leiden(adata, resolution=resolution, neighbors_key=neighbors_key, key_added='original_cluster', flavor="igraph", directed=False)
    else:
        sc.tl.louvain(adata, resolution=resolution, neighbors_key=neighbors_key, key_added='original_cluster')
        
    original_clusters = np.sort(adata.obs['original_cluster'].unique().astype(int))
    
    cluster_stats = []
    for i in original_clusters:
        idx_i = np.where(adata.obs['original_cluster'] == str(i))[0]
        
        if len(idx_i) > 1:
            block_ii_co = co_occurrence_sum[idx_i, :][:, idx_i].toarray()
            block_ii_samp = sampling_count[idx_i, :][:, idx_i].toarray()
            
            block_ii_samp[block_ii_samp == 0] = 1.0 
            block_ii_consensus = block_ii_co / block_ii_samp
            
            np.fill_diagonal(block_ii_consensus, np.nan)
            stability_i = np.nanmean(block_ii_consensus)
        else:
            stability_i = np.nan 

        max_confusion_i = 0.0
        for j in original_clusters:
            if i == j: continue
            idx_j = np.where(adata.obs['original_cluster'] == str(j))[0]
            
            block_ij_co = co_occurrence_sum[idx_i, :][:, idx_j].toarray()
            block_ij_samp = sampling_count[idx_i, :][:, idx_j].toarray()
            
            block_ij_samp[block_ij_samp == 0] = 1.0
            confusion_ij = np.mean(block_ij_co / block_ij_samp)
            
            if confusion_ij > max_confusion_i:
                max_confusion_i = confusion_ij
                
        score_i = stability_i - max_confusion_i
        cluster_stats.append(score_i)
        
    global_score = np.nanmean(cluster_stats)
    return resolution, global_score, len(original_clusters)


def run_parallel_stability(adata, min_res=0.1, max_res=2.0, step=0.2, use_rep=None, neighbors_key=None, clustering_algorithm='leiden', n_iterations=50, n_jobs=-1):
    sc.settings.verbosity = 0

    if use_rep is None:
        if 'X_pca' in adata.obsm:
            use_rep = 'X_pca'
        elif 'X_scVI' in adata.obsm:
            use_rep = 'X_scVI'
        else:
            # Search for custom namings (e.g., X_pca_harmony)
            pca_alts = [k for k in adata.obsm.keys() if 'X_pca' in k]
            if pca_alts:
                use_rep = pca_alts[0]
                print(f"Warning: 'X_pca' not found. Using '{use_rep}' embedding to build neighborhood graph for stability analysis.")
            else:
                raise ValueError("No valid latent space (X_pca, X_scVI) found for stability analysis. Please explicitly specify 'stability_use_rep'.")
    else:
        if use_rep not in adata.obsm:
            raise KeyError(f"Provided stability_use_rep '{use_rep}' not found in adata.obsm.")
        if use_rep in ['X_umap', 'X_tsne']:
            print(f"Warning: Computing neighborhood graph for stability on '{use_rep}'. Are you sure? (Advised: X_pca)")
    
    resolutions = np.arange(min_res, max_res, step)

    max_cores = multiprocessing.cpu_count()
    actual_jobs = max_cores if n_jobs == -1 else min(n_jobs, max_cores)
    num_cores = min(actual_jobs, len(resolutions))
    
    print(f"Performing {len(resolutions)} parallel bootstraps ({n_iterations} iterations each), using {num_cores}/{max_cores} CPU cores")
    
    # Stream results as a generator to allow real-time tqdm tracking
    results_gen = Parallel(n_jobs=num_cores, backend='loky', return_as="generator")(
        delayed(compute_stability_for_resolution)(res, adata, clustering_algorithm, neighbors_key, n_iterations, use_rep)
        for res in resolutions
    )
    results = list(tqdm(results_gen, total=len(resolutions), desc="Evaluating Stability"))
    
    df_results = pd.DataFrame(results, columns=['resolution', 'penalized_score', 'n_clusters'])
    raw_scores = df_results['penalized_score'].values
    
    # Identify indices of all local MAXIMA (peaks)
    maxima_indices = argrelextrema(raw_scores, np.greater, order=1)[0]
    
    if len(maxima_indices) > 0:
        # Get all resolutions that are valid peaks
        suggested_resolutions = df_results.iloc[maxima_indices]
        
        # Pick the highest peak among the non-trivial local maxima
        best_idx = suggested_resolutions['penalized_score'].idxmax()
        best_res = df_results.loc[best_idx, 'resolution']
        
        print(f"Detected {len(maxima_indices)} stable local maxima at resolutions: {suggested_resolutions['resolution'].tolist()}")
        print(f"Suggested Stability Resolution (Highest Peak): {best_res:.2f}")
    else:
        # Fallback to global maximum if no peaks exist
        best_res = df_results.loc[df_results['penalized_score'].idxmax(), 'resolution']
        print(f"No local maxima detected. Falling back to global maximum: {best_res:.2f}")
        
    return df_results, best_res


def plot_bachclue(df_score, best_res_score, score_metric, df_stab=None, best_res_stab=None):
    """Dynamically plots 1 or 2 panels depending on whether stability was computed."""
    
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
    """
    Main execution wrapper for BaCHClue clustering optimization.
    Prints result tables and returns a dictionary containing all generated DataFrames.
    """
    
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
        
        print(f"Optimization Summary for {score_value.upper()} & Stability")
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
        print(f"Optimization Summary for {score_value.upper()}")
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