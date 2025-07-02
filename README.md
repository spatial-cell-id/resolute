# BaCHClue: Bic and Calinski-Harabasz Score Guided Clustering

BaCHClue aims to find the optimal resolution parameter for clustering single-cell RNA-Seq data analyzed with _scanpy_, by calculating the BIC score or the Calinski-Harabasz score of each clustering, given a range of possible resolutions.

Here, the minimum of the BIC or Calinski-Harabasz score indicates the optimal resolution parameter.

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

Further details on BIC score can be found [here](https://www-sciencedirect-com.insb.bib.cnrs.fr/topics/mathematics/bayesian-information-criterion).

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

In the function, the Calinski-Harabasz score was calculated by using the function _calinski_harabasz_score()_ from _sklearn_ library, which takes as an input PCA or UMAP coordinates of each cell and a vector with the claster label of each cell.
For the purpose of plotting, -1*CH is defined to be the output, so then the lowest score indicates better-defined and well-separated clusters, as it happens with BIC score.

### Nedeed libraries
numpy, scanpy, sklearn


### Usage


```python
clustering_score(original_adata, score_value = 'bic', clustering_algorithm='leiden', dim_reduction = 'pca', min_res=0.1, max_res=2.0, step=0.1, plot=True)
```

To use the function, simply copy it from score_function.py and use in your own script.

#### Parameters:
* original_adata: the AnnData file containing the normalized and scaled data and dimensionality reductions (_PCA_ and _UMAP_ or _tsne_)
* score_value = the chosen score to evaluate the different clusterings. Possible choices: **bic**,**calinski**. Default: **'calinski'**
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
