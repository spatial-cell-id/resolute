import scanpy as sc
import time
import bachclue as bc

print("Loading data...")


adata = sc.read_h5ad('datasets/neurons_2000/processed_data.h5ad')


print("Data loaded")


start_time = time.time()
a = bc.clustering_score(adata, score_value='calinski', min_res=0.1, max_res=2, step=0.1, plot=True, dim_reduction='umap')
end_time = time.time()
print(f"OPTIMIZED, MULTIPROCESSING: Execution time: {end_time - start_time:.2f} seconds")
