version 1.0

workflow cluster_robustness {
    input {
    	String output_directory
        File anndata
        Array[Float] resolutions
        Int num_iter
        Float subset_ratio = 0.9
        #general parameters
        Int cpu = 8
        String memory = "64G"
        String docker = "mparikhbroad/cluster_robustness:latest"
        Int preemptible = 2
    }

    String output_directory_stripped = sub(output_directory, "/+$", "")

    scatter(resolution in resolutions) {
        call run_cluster_robustness {
            input:
                anndata = anndata,
                resolution = resolution,
                subset_ratio = subset_ratio,
                num_iter = num_iter,
                cpu=cpu,
                memory=memory,
                docker=docker,
                preemptible=preemptible
        }
    }

    call compile_iterations {
        input:
            output_dir = output_directory_stripped,
            rand_index_score_files = run_cluster_robustness.rand_index_scores_file,
            silhouette_score_files = run_cluster_robustness.silhouette_scores_file,
            num_clusters_files = run_cluster_robustness.num_clusters_file,
            resolutions = resolutions,
            num_iter = num_iter,
            cpu=cpu,
            memory=memory,
            docker=docker,
            preemptible=preemptible
    }

    output {
        File rand_index_score_plot_pdf = compile_iterations.rand_index_score_plot_pdf
        File silhouette_score_plot_pdf = compile_iterations.silhouette_score_plot_pdf
        File num_clusters_plot_pdf = compile_iterations.num_clusters_plot_pdf
        File rand_index_score_table_csv = compile_iterations.rand_index_score_table_csv
        File silhouette_score_table_csv = compile_iterations.silhouette_score_table_csv
        File num_clusters_table_csv = compile_iterations.num_clusters_table_csv
    }
}

task run_cluster_robustness {

    input {
        File anndata
        Float resolution
        Float subset_ratio
        Int num_iter
        String memory
        Int cpu
        String docker
        Int preemptible
    }

    command {
        set -e

        python << CODE
        import os
        import numpy as np
        import pandas as pd
        import scanpy as sc
        import scanpy.external as sce
        import scipy as sp
        from sklearn.model_selection import train_test_split
        from sklearn.metrics.cluster import adjusted_rand_score
        from sklearn.metrics import silhouette_score
        from tqdm import tqdm
        import matplotlib.pyplot as plt

        adata = sc.read_h5ad('~{anndata}')
        sc.tl.leiden(adata, resolution=~{resolution})
        rand_score_arr = []
        silhouette_score_arr = []
        num_clusters_arr = []
        orig_silhouette_score = silhouette_score(adata.obsm['X_pca'], adata.obs['leiden'])
        silhouette_score_arr.append(orig_silhouette_score)
        orig_num_clusters = len(adata.obs.leiden.unique())
        num_clusters_arr.append(orig_num_clusters)
        for j in tqdm(range(0, ~{num_iter}), desc='iteration'):
            #get subset and save clustering solution from full dataset
            train, test = train_test_split(adata.obs, train_size=~{subset_ratio}, random_state=(j), stratify=adata.obs[['leiden']])
            temp = adata[train.index].copy()
            temp.obs['original_leiden'] = temp.obs['leiden']
            #rerun HVG, PCA, and leiden clustering on subset
            sc.pp.highly_variable_genes(temp, min_mean=0.0125, max_mean=3, min_disp=0.5)
            sc.tl.pca(temp, svd_solver='arpack')
            sc.pp.neighbors(temp, n_neighbors=10, n_pcs=40)
            sc.tl.leiden(temp, resolution=~{resolution})
            #calculate adjusted rand score between original cluster labels and new subset cluster labels
            rand_score_arr.append(adjusted_rand_score(temp.obs.original_leiden, temp.obs.leiden))
            silhouette_score_arr.append(silhouette_score(temp.obsm['X_pca'], temp.obs['leiden']))
            num_clusters_arr.append(len(temp.obs.leiden.unique()))
        np.save('rand_score.~{resolution}.npy', rand_score_arr, allow_pickle=False)
        np.save('silhouette_score.~{resolution}.npy', silhouette_score_arr, allow_pickle=False)
        np.save('num_clusters.~{resolution}.npy', num_clusters_arr, allow_pickle=False)
        CODE
        
    }

    output {
        File rand_index_scores_file = "rand_score.~{resolution}.npy"
        File silhouette_scores_file = "silhouette_score.~{resolution}.npy"
        File num_clusters_file = "num_clusters.~{resolution}.npy"
    }

    runtime {
        docker: docker
        memory: memory
        bootDiskSizeGb: 12
        disks: "local-disk " + ceil(size(anndata, "GB")*2) + " HDD"
        cpu: cpu
        preemptible: preemptible
    }

}

task compile_iterations {

    input {
        String output_dir
        Array[File] rand_index_score_files
        Array[File] silhouette_score_files
        Array[File] num_clusters_files
        Array[Float] resolutions
        Int num_iter
        String memory
        Int cpu
        String docker
        Int preemptible
    }

    command <<<
        set -e

        mkdir -p outputs

        python << CODE
        import os
        import numpy as np
        import pandas as pd
        import scipy as sp
        import seaborn as sns
        import matplotlib.pyplot as plt

        list_of_rand_score_files = ["~{sep='","' rand_index_score_files}"]
        list_of_silhouette_score_files = ["~{sep='","' silhouette_score_files}"]
        list_of_num_clusters_files = ["~{sep='","' num_clusters_files}"]
        list_of_rand_score_lists = []
        list_of_silhouette_score_lists = []
        list_of_num_clusters_lists = []
        for score_file in list_of_rand_score_files:
            temp = np.load(score_file)
            list_of_rand_score_lists.append(temp)
        for score_file in list_of_silhouette_score_files:
            temp = np.load(score_file)
            list_of_silhouette_score_lists.append(temp)
        for score_file in list_of_num_clusters_files:
            temp = np.load(score_file)
            list_of_num_clusters_lists.append(temp)
        resolution_list = [~{sep=", " resolutions}]
        rand_score_df = pd.DataFrame(list_of_rand_score_lists, index=[str(i) for i in resolution_list], columns=[str(i) for i in range(0, ~{num_iter})])
        rand_score_df.to_csv('outputs/rand_index_score_table.csv')
        silhouette_score_df = pd.DataFrame(list_of_silhouette_score_lists, index=[str(i) for i in resolution_list], columns=[str(i) for i in range(0, ~{num_iter}+1)])
        silhouette_score_df.to_csv('outputs/silhouette_score_table.csv')
        num_clusters_df = pd.DataFrame(list_of_num_clusters_lists, index=[str(i) for i in resolution_list], columns=[str(i) for i in range(0, ~{num_iter}+1)])
        num_clusters_df.to_csv('outputs/num_clusters_table.csv')

        plt.figure(figsize=(len(resolution_list), 10))
        ax = sns.boxplot(data=rand_score_df.T)
        ax.set(title=f'Rand Index Scores per Leiden Resolution\n({~{num_iter}} iterations)', xlabel='Leiden Resolution', ylabel='Rand Index Score', ylim=(0,1))
        ax.figure.savefig('outputs/rand_index_score_plot.pdf')

        plt.figure(figsize=(len(resolution_list), 10))
        ax = sns.boxplot(data=silhouette_score_df.T)
        ax.set(title=f'Silhouette Scores per Leiden Resolution\n({~{num_iter}} iterations)', xlabel='Leiden Resolution', ylabel='Silhouette Score', ylim=(-1,1))
        sns.swarmplot(ax=ax, data=silhouette_score_df[["0"]].T, color='black', size=10)
        ax.figure.savefig('outputs/silhouette_score_plot.pdf')

        plt.figure(figsize=(len(resolution_list), 10))
        ax = sns.boxplot(data=num_clusters_df.T)
        ax.set(title=f'Num Clusters per Leiden Resolution\n({~{num_iter}} iterations)', xlabel='Leiden Resolution', ylabel='Num Clusters')
        sns.swarmplot(ax=ax, data=num_clusters_df[["0"]].T, color='black', size=10)
        ax.figure.savefig('outputs/num_clusters_plot.pdf')

        CODE

        gsutil -m rsync -r outputs ~{output_dir}

    >>>

    output {
        File rand_index_score_plot_pdf = 'outputs/rand_index_score_plot.pdf'
        File silhouette_score_plot_pdf = 'outputs/silhouette_score_plot.pdf'
        File num_clusters_plot_pdf = 'outputs/num_clusters_plot.pdf'
        File rand_index_score_table_csv = 'outputs/rand_index_score_table.csv'
        File silhouette_score_table_csv = 'outputs/silhouette_score_table.csv'
        File num_clusters_table_csv = 'outputs/num_clusters_table.csv'
    }

    runtime {
        docker: docker
        memory: memory
        bootDiskSizeGb: 12
        disks: "local-disk " + ceil(size(rand_index_score_files, "GB")*4) + " HDD"
        cpu: cpu
        preemptible: preemptible
    }

}