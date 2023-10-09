def clustering_score(original_adata, score_value = 'bic', clustering_algorithm='leiden', dim_reduction = 'pca', min_res=0.1, max_res=2.0, step=0.1, plot=True):
    #calinski_harabasz
    import numpy as nu
    import scanpy as sc
    import pandas as pd
    from sklearn.metrics import calinski_harabasz_score

    sc.settings.verbosity = 0

    if dim_reduction == 'pca':
        dim_reduction = 'X_pca'
    elif dim_reduction == 'umap':
        dim_reduction = 'X_umap'
    else:
        print('please choose pca or umap as dimensionality reduction')
        exit

    res = list(nu.arange(min_res,max_res,step))
    score = []
    n_clus = []

    for r in res:
        index = res.index(r)
        adata = original_adata.copy()
        string_r = str(r)
        clustering_name = '%s_res%s' %(clustering_algorithm,string_r)
        print('Clustering by using the resolution %.2f, step %i of %i' %(r,index+1,len(res)))
        if clustering_algorithm == 'leiden':
            sc.tl.leiden(adata, key_added="%s_res%s" %(clustering_algorithm,string_r), resolution=r)
        elif clustering_algorithm == 'louvain':
            sc.tl.louvain(adata, key_added="%s_res%s" %(clustering_algorithm,string_r), resolution=r)
        else:
            print('please choose louvain or leiden as clustering_algorithm')
            exit
        if score_value == 'bic':
            score_name = 'BIC score'
            n_points = len(adata.obs[clustering_name])
            n_clusters = len(set(adata.obs[clustering_name]))
            n_dimensions = adata.obsm[dim_reduction].shape[1]

            n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1

            loglikelihood=0
            for cluster_id in set(adata.obs[clustering_name]):
                cluster_mask = (adata.obs[clustering_name] == cluster_id)
                X_cluster = adata.obsm[dim_reduction][cluster_mask]

                n_points_cluster = len(X_cluster)
                centroid = nu.mean(X_cluster, axis=0)
                variance = nu.sum((X_cluster - centroid) ** 2) / (len(X_cluster) - 1)
                loglikelihood += \
                  n_points_cluster * nu.log(n_points_cluster) \
                  - n_points_cluster * nu.log(n_points) \
                  - n_points_cluster * n_dimensions / 2 * nu.log(2 * nu.pi * variance) \
                  - (n_points_cluster - 1) / 2

            bic = loglikelihood - (n_parameters / 2) * nu.log(n_points)
            bic = -2*bic
            score.append(bic)
            n_clus.append(n_clusters)

            del adata.obs[clustering_name]
        
        elif score_value == 'calinski':
            score_name = 'Calinski-Harabasz score'
            n_clusters = len(set(adata.obs[clustering_name]))
            n_clus.append(n_clusters)
            score.append(-1*calinski_harabasz_score(adata.obsm[dim_reduction], adata.obs[clustering_name]))
            

    df = pd.DataFrame()
    df[score_name] = score
    df['resolution'] = res
    df['n_clus'] = n_clus

    if plot==True:
        df.plot(x='resolution',y=score_name)
    print('\n')

    return df, res[nu.argmin(score)]