---
title: 'RESOLUTE: A python tool for optimal resolution selection in single-cell RNA-Seq and Spatial Transcriptomcis data clustering'
tags:
  - Python
  - scRNA-Seq data analysis
  - BIC and Calinski-Harabasz score
  - Unsupervised clustering

authors:
  - name: Marco Uderzo
    orcid: 0000-0000-0000-0000
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
  - name: Sergio Sarnataro
    orcid: 0000-0002-6001-9892
    corresponding: true
    affiliation: 1
affiliations:
 - name: École Normale Supérieure (ENS), Lyon, France
   index: 1
 - name: Institut de Génomique Fonctionnelle de Lyon (IGFL), UMR5242, Lyon, France
   index: 2
 - name: Centre National de la Recherche Scientifique
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Clustering is a fundamental step in single-cell RNA sequencing (scRNA-seq) and spatial transcriptomics (ST) analysis, as it identifies cell subgroups with similar transcriptional profiles, facilitating cell-type annotation and the discovery of novel cell types. However, common clustering algorithms require users to manually specify a resolution parameter, which significantly affects the number of clusters and the resulting granularity of the analysis. This process introduces a subjective component that can hinder reproducibility. RESOLUTE (Robust Evaluation of Single-cell Optimal Leiden resolUtion on Topological Embeddings) is a Python package designed to automate the selection of the optimal clustering resolution. By leveraging established graph-theory and statistical metrics, such as the Bayesian Information Criterion (BIC) and the Calinski-Harabasz index, RESOLUTE provides an objective, reproducible, and scalable framework for bioinformaticians to determine the most relevant clustering resolution in their single-cell data. Notably, RESOLUTE is fully compatible with the scVerse ecosystem.

# Statement of need

Clustering is a critical step in single-cell RNA sequencing and spatial transcriptomics data analysis. While clustering is theoretically an unsupervised process intended to be only data-driven, the selection of the resolution parameter often introduces a strong subjective human component. This parameter dictates the trade-off between over-clustering, which can lead to the identification of artificial sub-populations, and under-clustering, where biologically distinct cell types are merged.

In practice, researchers often perform resolution selection by manually adjusting parameters to match prior biological knowledge or by relying on default settings provided by standard clustering tools. In both scenarios, the resulting clustering output is potentially biased and lacks statistical rigor.

RESOLUTE addresses this challenge by providing an objective framework for resolution selection. It systematically evaluates potential clustering resolutions post-hoc using robust statistical metrics, specifically the Bayesian Information Criterion (BIC) and the Calinski-Harabasz score. Furthermore, RESOLUTE assesses clustering stability through a bootstrapping procedure, offering bioinformaticians a mathematically and statistically grounded starting point for their downstream analyses. By doing that, RESOLUTE minimizes subjective bias, and potentially enhances the reproducibility of single-cell data analysis workflows.

# State of the field  

Several tools exist for cluster optimization, such as clustree \cite{clustree} in the R ecosystem. While clustree is excellent for visualizing cluster stability across resolutions, it is primarily an interactive visualization tool and does not provide an automated "optimal choice" recommendation for high-throughput automated pipelines.
Another prominent state-of-the-art tool is MultiK \cite{liu2021multik} , which utilizes a multi-scale consensus clustering approach to identify stable cluster numbers ($K$). MultiK iteratively subsamples the data matrix and runs clustering algorithm across a list of resolutions. It then evaluates stability using the Proportion of Ambiguous Clustering (PAC) metric and test whether adjacent clusters represent distinct multivariate normal distributions. 

While MultiK successfully uncovers stable hierarchical levels of cell organization, it relies on the construction of dense, pairwise cell-by-cell consensus matrices across multiple subsampling iterations. 
This introduces a severe memory and computational bottleneck ($O(N^2)$ space complexity), making it virtually prohibitive scaling into hundreds of thousands of cells or high-throughput spatial atlases. To address these scalability constraints, \texttt{RESOLUTE} can identify optimal resolutions in a highly efficient manner, by using structural and variance-based metrics (BIC and Calinski-Harabasz) applied directly to low-dimensional embeddings, bypassing iterative resampling altogether when processing ultra-large datasets. 
When topological robustness is queried, \texttt{RESOLUTE}'s bootstrap module performs parallelized neighborhood graph reconstruction using \texttt{joblib}, avoiding the prohibitive $N \times N$ memory footprint of classical consensus clustering approaches.

We chose to build RESOLUTE rather than contributing to existing R-based packages because the current standard for single-cell analysis in many research institutions is increasingly centered on the Scanpy (Python) framework. RESOLUTE fills a specific gap by offering an integrated, Python solution that functions natively within the AnnData object structure. Its unique contribution is the automated, quantitative ranking of resolutions based on the consistency of the biological signal, rather than merely relying on visual inspection.

# Software design

_RESOLUTE_ is implemented as an open-source Python package engineered to integrate seamlessly with contemporary single-cell and spatial transcriptomics analysis pipelines. It is built natively on top of the _Scanpy_ framework \cite{wolf2018scanpy} and interfaces directly with core scientific computing libraries including _NumPy_, _scikit-learn_, and _joblib_. 

A core principle of the _RESOLUTE_ software design is object preservation and functional purity. Unlike many single-cell utilities that mutate the primary data structure by adding transient categorical labels or unstructured arrays, _RESOLUTE_ treats the input _AnnData_ object as read-only. The execution pipeline returns an isolated, structured dictionary containing detailed results DataFrames.
This decoupled architecture protects user workflows from downstream state-mutation side effects and ensures compliance with reproducible data science practices.

### Algorithmic Workflow and Mathematical Optimization
The software architecture follows a modular, dual-layered optimization strategy across a continuous user-defined resolution vector $(\mathcal{R} = [res_{min}, res_{max}, \Delta res])$. The execution flow is divided into three distinct execution steps:

- Graph Partitioning Sweep: For each resolution $\gamma \in \mathcal{R}$, data points are partitioned into communities using a Leiden algorithm built upon a pre-computed neighborhood graph stored within the _AnnData_ object.
    
- Geometric Scoring Engine: Following community detection, the software assess cluster properties by using two primary geometric cost functions:
  - Bayesian Information Criterion (BIC): Evaluates the trade-off between the statistical model likelihood and the degree of freedom penalties associated with increasing cluster numbers ($k$).
  - Calinski-Harabasz (CH) Score: Evaluates the ratio of between-cluster variance to within-cluster variance. Because the traditional CH score maximizes with cluster quality, _RESOLUTE_ minimizes an inverted function $(-1 \times CH)$ to maintain structural alignment with the BIC optimization engine.

Importantly, the user can decouple the geometric evaluation space _use_rep_, e.g., _X_umap_ or _X_scVI_ from the initial topological space.
    
- Topological Bootstrap Module: Operating independently of the BIC and Calinski-Harabasz score, an optional bootstrap stability analysis can be performed (**compute_stability=True**). This module measures the structural robustness of the selected communities against local perturbations. For each resolution parameter, the software iteratively draws random subsets of cells, reconstructs the localized neighborhood graph on a designated latent representation _stability_use_rep_, typically _X_pca_, and runs community detection to assess topological stability across iterations.

### Computational Performance
To accommodate the high-throughput scale of huge spatial and single-cell datasets, _RESOLUTE_ incorporates a highly parallelized execution loop. Rather than sequentially executing the resolution sweep and iterative bootstrapping, which introduces severe processing bottlenecks, the software delegates compute threads across all available CPU cores using native multi-processing via _joblib_.

By evaluating the geometric metrics directly on low-dimensional matrix representations (e.g., PCA or UMAP embeddings) rather than storing large matrices, _RESOLUTE_ operates with highly efficient memory and time complexity scaling. This allows the software to compute multi-scale parameters rapidly without inducing Out-Of-Memory (OOM) faults on typical bioinformatics machines, establishing it as a highly scalable solution for large cellular atlases.

# Research impact statement

Write something regarding the intensive use at ENS of Lyon?

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# AI usage disclosure
During the preparation of this work, the authors utilized generative artificial intelligence technologies (specifically, Google's Gemini large language model) for two distinct purposes:
-  Manuscript Preparation: The AI was used to assist in structuring, drafting, and refining the language of the manuscript, including the synthesis of comparative literature and software design descriptions.
- AI-guided suggestions were utilized during the engineering of the _RESOLUTE_ Python package. Specifically, the AI assisted in drafting the iterative sampling scripts for the topological bootstrap stability module and optimizing the parallel execution framework (_joblib_) for the Bayesian Information Criterion (BIC) and Calinski-Harabasz  scoring metrics.

The authora thoroughly reviewed, tested, and edited all generated text and underlying source code. The authors take full responsibility for the scientific accuracy, code integrity, and final content of this publication.

# Acknowledgements


# References
