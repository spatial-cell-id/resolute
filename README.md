# RESOLUTE: Robust Evaluation of Single-cell Optimal Leiden resolUtion on Topological Embeddings

RESOLUTE is a Python package designed for selecting the optimal clustering resolution in Scanpy-based single-cell and spatial transcriptomics analyses.

In high-dimensional transcriptomic data, clustering algorithms (like Leiden or Louvain) rely heavily on a user-defined resolution parameter, which dictates the granularity of the resulting cell populations. Traditionally, selecting this resolution involves manual, subjective trial-and-error. RESOLUTE eliminates this ambiguity by evaluating multiple clustering resolutions across a defined range and computing statistical validation metrics. This allows researchers to identify biologically meaningful cluster structures based on quantitative, reproducible criteria.

## Core Metrics & Theoretical Foundation
RESOLUTE optimizes the resolution parameter by seeking the mathematical minimum of specific scoring functions. It currently supports two primary geometric metrics, alongside a topological stability evaluation.

### BIC score
The BIC (Bayesian Information Criterion) score is a statistical measure used  to assess the goodness of fit of a statistical model. It is often used in the context of model selection among a set of candidate models.

The BIC score derives from Bayesian probability theory and is based on the likelihood function of the data given a particular model. It balances the trade-off between model complexity (the number of parameters in the model) and the goodness of fit to the data. The goal is to find a model that fits the data well while penalizing overly complex models that may overfit the data.

In the function, the BIC score was manually implemented by using the formula:

```math
BIC = -2*log(L) + k*log(N)
```

Where:
- L is the likelihood of the data given the model;
- k is the number of parameters in the model;
- N is the number of data points.

Interpretation: The algorithm searches for local or global minima across the resolution range. The lowest BIC score indicates the optimal balance of biological signal and cluster compactness.

### Calinski-Harabasz score
The Calinski-Harabasz (CH) score is a metric used for evaluating the quality of clusters in unsupervised machine learning, particularly in clustering analysis. It aims to determine the optimal number of clusters. The CH score is based on the ratio of the between-cluster variance to the within-cluster variance. It measures the compactness of clusters (small within-cluster variance) relative to their separation (large between-cluster variance). A higher CH score indicates better-defined and well-separated clusters.

The formula for calculating the Calinski-Harabasz score is as follows:
```math
CH = (B / (k - 1)) / (W / (n - k))
```

Where:
- CH is the Calinski-Harabasz score;
- B is the between-cluster variance, which measures the variance between cluster centroids;
- k is the number of clusters;
- W is the within-cluster variance, which measures the variance within each cluster;
- n is the total number of data points.

By default, a higher standard CH score indicates better-defined and well-separated clusters. However, to maintain programmatic consistency with the BIC optimization (where the goal is minimization), RESOLUTE calculates the CH score using sklearn and inherently outputs $-1 \times CH$.

Interpretation: Just like the BIC score, the algorithm identifies the optimal resolution by finding the deepest dip (minimum) in the transformed Calinski-Harabasz curve.


### Bootstrap Stability Analysis
In addition to geometric metrics, RESOLUTE can perform iterative subsetting (bootstrapping) to evaluate the topological stability of the clusters at each resolution. By recalculating the neighborhood graph on subsets of the data, this feature ensures the chosen resolution is robust against data perturbation and not an artifact of the specific embedding.


### Dependencies
numpy, scanpy, sklearn, joblib


### Usage

Install by pip:

```bash
pip install resolute 
```

RESOLUTE integrates smoothly into standard Scanpy workflows. It is highly recommended to run your standard preprocessing pipeline (PCA, neighbor graph computation, UMAP) before executing the tool.

#### Example Workflow

```python
import scanpy as sc
import resolute as rs

# Load and preprocess your data
adata = sc.read_h5ad('adata.h5ad')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata)
sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.umap(adata)

# Run RESOLUTE Optimization
results = bc.run_resolute(
    adata=adata, 
    score_value='calinski', 
    use_rep='X_umap',           # Embedding for evaluating cluster geometry
    stability_use_rep='X_pca',  # Embedding for rebuilding topological graphs
    compute_stability=True, 
    min_res=0.1,
    max_res=2.0,
    step=0.1,
    n_iterations=50
)

# Access Best Resolutions
best_geometric_res = results['best_score_res']
best_stable_res = results['best_stability_res']
print(f"Optimal clustering resolution: {best_geometric_res}")
```


#### Parameters:

- `adata` (AnnData): The annotated data matrix.

- `score_value` (str, default='bic') : The geometric scoring metric to optimize. Options are 'bic' or 'calinski'.

- `clustering_algorithm` (str, default='leiden'): The algorithm to use for partitioning. Options are 'leiden' or 'louvain'.

- `use_rep` (str, default='X_pca'): The dimensional embedding used to calculate the geometric scores (e.g., 'X_pca', 'X_umap', 'X_scVI').

- `stability_use_rep` (str, optional): The representation used to iteratively reconstruct the neighborhood graph during bootstrap stability analysis. If None, it defaults to use_rep. Note: It is highly recommended to use a robust latent space (like X_pca) rather than a 2D projection (like X_umap) for this parameter.

- `min_res` (float, default=0.1): The starting resolution for the optimization sweep.

- `max_res` (float, default=2.0): The maximum resolution for the optimization sweep.

- `step` (float, default=0.1): The step size between evaluated resolutions.

- `compute_stability` (bool, default=True): Whether to run the bootstrap stability analysis alongside the geometric scoring.

- `n_iterations` (int, default=50): The number of bootstrap subsampling iterations per resolution.

- `n_jobs` (int, default=-1): The number of parallel jobs to run. -1 uses all available CPU cores.

- `plot` (bool, default=True): If True, automatically generates and displays the optimization curves, highlighting local extrema and the suggested best resolution.

#### Returns:

A dictionary containing:

- `'score_results'`: A pandas DataFrame containing the geometric scores and cluster counts for each resolution.

- `'best_score_res'`: The float value of the optimal geometric resolution.

- `'stability_results'`: A pandas DataFrame containing stability scores (if computed).

- `'best_stability_res'`: The float value of the most topologically stable resolution.



Note that RESOLUTE does not add any clustering metadata into the original AnnData file.

### Example (Deprecated - to rewrite):
The notebook _tests/score_function_3kPBMC.ipynb_ contains and example usage of the function. Data consist of 3k PBMCs from a Healthy Donor and are freely available from 10x Genomics, and can be download as follows from a terminal:
```bash
wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz -O data/pbmc3k_filtered_gene_bc_matrices.tar.gz
```
Alternatively, they can be downloaded [here](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k).

Data were preprocessed and analyzed by following this [tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html).

The obtained optimum resolution was 1.0, corresponding to 7 different clusters, as expected from [here](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html).
