# Optimum resolution


This function aims to find the optimum resolution parameter for clustering single-cell RNA-Seq data analyzed with _scanpy_, by calculating the BIC score of each clustering.

It has been inspired by this [read](https://towardsdatascience.com/are-you-still-using-the-elbow-method-5d271b3063bd) and then adapted to the single-cell analysis in scanpy.

Here, the minimum of the BIC score indicates the optimum resolution parameter.

### Nedeed libraries



### Usage
```python
bic_score(original_adata, clustering_algorithm='leiden', dim_reduction = 'pca', min_res=0.1, max_res=2.0, step=0.1, plot=True)
```

To use the function, simply copy it from score_function.py and use in your own script.

#### Parameters:
* original_adata: the AnnData file containing the normalized and scaled data and dimensionality reductions (_PCA_ and _UMAP_ or _tsne_)
* clustering_algorithm: the algorithm to use to test the different clustering (string). Possible choices: **leiden**,**louvain**. Default: **'leiden'**.
* dim_reduction: coordinates to use for calculating the BIC score (string). Possible choices: **pca**, **umap**. Default: **'pca'**.
* min_res: minimum resolution to test (float). Default: **0.1**.
* max_res: maximum resolution to test (float). Default: **2.0**.
* step: step size for resolutions to be tested between _min_res_ and _max_res_ (float). Default: **0.1**
* plot: Whether to plot or not the BIC score as function of the resolution (Boolean). Default: **True**.

#### Outputs:
The function returns two outputs:
* A pandas DataFrame with 3 columns, i.e. BIC score, resolution and corresponding number of clusters;
* the optimum of the resolution, that can be used for further analysis.

Note that the function does not add any clustering metadata into the original AnnData file.

### Example:
The notebook _score_function.ipynb_ contains and example usage of the function. Data consist of 3k PBMCs from a Healthy Donor and are freely available from 10x Genomics, and can be download as follows from a terminal:
```bash
wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz -O data/pbmc3k_filtered_gene_bc_matrices.tar.gz
```
Alternatively, they can be downloaded [here](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k).

Data were preprocessed and analyzed by following this [tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html).

The obtained optimum resolution was 1.0, corresponding to 7 different clusters, as expected from [here](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html).
