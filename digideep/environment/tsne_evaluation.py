from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def tsne_evaluation(storage):
    inputs = storage.input_batch
    obs_clusters = storage.obs_cluster_batch
    act_clusters = storage.action_cluster_batch
    actions = storage.action_batch
    hidden_action = storage.hidden_action_batch

    act_clusters_labels = act_clusters.argmax(dim=1)

    X = hidden_action
    y = act_clusters_labels
    feat_cols = ['dimension' + str(i) for i in range(X.shape[1])]
    df_subset = pd.DataFrame(X, columns=feat_cols)
    df_subset['y'] = y
    df_subset['label'] = df_subset['y'].apply(lambda i: str(i))

    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))


    X, y = None, None
    print('Size of the dataframe: {}'.format(df_subset.shape))

    # 결과 재생산을 위해
    np.random.seed(42)

    data_subset = df_subset[feat_cols].values
    print(data_subset.shape)

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=10000)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",hue="y",data=df_subset,legend="full",alpha=0.3, palette= 'deep')
    plt.show()

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)

    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    plt.figure(figsize=(16, 10))
    sns.scatterplot(x="pca-one", y="pca-two",hue="y",palette='deep',data=df,legend="full",alpha=0.3)

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df["pca-one"],
        ys=df["pca-two"],
        zs=df["pca-three"],
        c=df["y"]
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()


    import pdb
    pdb.set_trace()