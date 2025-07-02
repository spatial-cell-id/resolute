import numpy as np
import scanpy as sc
import pandas as pd
import multiprocessing
from sklearn.metrics import calinski_harabasz_score
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def compute_clustering_score(r, original_adata, score_value, clustering_algorithm, dim_reduction):
    """Compute clustering score for a given resolution."""
    adata = original_adata.copy()
    clustering_name = f"{clustering_algorithm}_res{r:.2f}"

    # Perform clustering
    if clustering_algorithm == 'leiden':
        sc.tl.leiden(adata, key_added=clustering_name, resolution=r)
    elif clustering_algorithm == 'louvain':
        sc.tl.louvain(adata, key_added=clustering_name, resolution=r)
    else:
        raise ValueError("Please choose 'leiden' or 'louvain' as clustering_algorithm")

    # Compute score
    n_clusters = adata.obs[clustering_name].nunique()
    n_points = len(adata.obs[clustering_name])
    n_dimensions = adata.obsm[dim_reduction].shape[1]

    if score_value == 'bic':
        score_name = "BIC score"
        n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1
        loglikelihood = sum(
            len(X_cluster) * np.log(len(X_cluster))
            - len(X_cluster) * np.log(n_points)
            - (len(X_cluster) * n_dimensions / 2) * np.log(2 * np.pi * variance)
            - (len(X_cluster) - 1) / 2
            for cluster_id in adata.obs[clustering_name].unique()
            if (X_cluster := adata.obsm[dim_reduction][(adata.obs[clustering_name] == cluster_id).values]).shape[0] > 1
            and (variance := np.var(X_cluster, axis=0).sum()) > 0
        )
        score_value = -2 * (loglikelihood - (n_parameters / 2) * np.log(n_points))

    elif score_value == 'calinski':
        score_name = "Calinski-Harabasz score"
        score_value = -1 * calinski_harabasz_score(adata.obsm[dim_reduction], adata.obs[clustering_name])
    
    return r, score_value, n_clusters


def clustering_score(original_adata, score_value='bic', clustering_algorithm='leiden',
                      dim_reduction='pca', min_res=0.1, max_res=2.0, step=0.1, plot=True):
    """Compute clustering scores over a range of resolutions in parallel using Joblib."""
    sc.settings.verbosity = 0  # Suppress Scanpy verbosity

    # Validate dimensionality reduction
    if dim_reduction == 'pca':
        dim_reduction = 'X_pca'
    elif dim_reduction == 'umap':
        dim_reduction = 'X_umap'
    else:
        raise ValueError("Please choose 'pca' or 'umap' as dim_reduction")

    # Generate resolution range
    resolutions = np.arange(min_res, max_res, step)
    
    # Determine number of available CPU cores
    num_cores = min(multiprocessing.cpu_count(), len(resolutions))
    print(f"Performing {len(resolutions)} parallel clusterings")
    print(f"Using {num_cores}/{multiprocessing.cpu_count()} CPU cores")
    
    print(f"Starting parallel computation with Joblib using {num_cores} cores...")
    
    results = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(compute_clustering_score)(r, original_adata, score_value, clustering_algorithm, dim_reduction) 
        for r in resolutions
    )

    df = pd.DataFrame(results, columns=['resolution', score_value, 'n_clusters'])

    # Return DataFrame and best resolution
    best_res = df.iloc[df[score_value].idxmin()]['resolution']
    print(f"\nBest resolution: {best_res:.2f}")

    # Plot results
    if plot:
        df.plot(x='resolution', y=score_value)
        plt.show()
    
    return df, best_res