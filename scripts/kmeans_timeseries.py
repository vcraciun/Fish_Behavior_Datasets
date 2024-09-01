import numpy
import matplotlib.pyplot as plt
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from sklearn.model_selection import train_test_split 
from tslearn.utils import to_time_series_dataset

def euclidian_kmeans(X_train, seed, max_clusters):
    print("Euclidean k-means")
    sil_score_max = -1
    best_n_clusters = 0
    for n_clusters in range(2, max_clusters):
        km = TimeSeriesKMeans(n_clusters, verbose=False, random_state=seed, max_iter=100)                
        y_pred = km.fit_predict(X_train)
        sil_score = silhouette_score(X_train, y_pred)
        print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
    print(best_n_clusters)

    km = TimeSeriesKMeans(best_n_clusters, verbose=False, random_state=seed, max_iter=100)        
    y_pred = km.fit_predict(X_train)
    print(km.labels_)
    plt.subplots(figsize =(15, 6))
    for yi in range(best_n_clusters):
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(km.cluster_centers_[yi].ravel(), label = f"Cluster {yi+1}")
    plt.title("Euclidean $k$-means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"euclidian_kmeans_{best_n_clusters}.png", bbox_inches='tight', dpi=600)

def dba_kmeans(X_train, seed, max_clusters):
    print("DBA k-means")
    sil_score_max = -1
    best_n_clusters = 0
    for n_clusters in range(2, max_clusters):
        dba_km = TimeSeriesKMeans(n_clusters, n_init=2, metric="dtw", verbose=False, max_iter_barycenter=10, random_state=seed, max_iter=100)
        y_pred = dba_km.fit_predict(X_train)
        sil_score = silhouette_score(X_train, y_pred)
        print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
    print(best_n_clusters)

    dba_km = TimeSeriesKMeans(best_n_clusters, n_init=2, metric="dtw", verbose=False, max_iter_barycenter=10, random_state=seed)
    y_pred = dba_km.fit_predict(X_train)
    print(dba_km.labels_)
    plt.subplots(figsize =(15, 6))
    for yi in range(best_n_clusters):
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(dba_km.cluster_centers_[yi].ravel(), label = f"Cluster {yi+1}")
    plt.title("DBA $k$-means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"DBA_kmeans_{best_n_clusters}.png", bbox_inches='tight', dpi=600)

def soft_dtw_kmeans(X_train, seed):
    print("Soft-DTW k-means")
    sil_score_max = -1
    best_n_clusters = 0
    for n_clusters in range(2, 10):
        sdtw_km = TimeSeriesKMeans(n_clusters, metric="softdtw", metric_params={"gamma": .01}, verbose=False, random_state=seed, max_iter=100)
        y_pred = sdtw_km.fit_predict(X_train)
        sil_score = silhouette_score(X_train, y_pred)
        print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
    print(best_n_clusters)

    sdtw_km = TimeSeriesKMeans(best_n_clusters, metric="softdtw", metric_params={"gamma": .01}, verbose=False, random_state=seed)
    y_pred = sdtw_km.fit_predict(X_train)
    print(sdtw_km.labels_)
    plt.subplots(figsize =(15, 6))
    for yi in range(best_n_clusters):
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(sdtw_km.cluster_centers_[yi].ravel(), label = f"Cluster {yi+1}")
    plt.title("Soft-DTW $k$-means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Soft_DTW_kmeans_{best_n_clusters}.png", bbox_inches='tight', dpi=600)

seed = 0
numpy.random.seed(seed)
df = pd.read_csv("zebra_all_features.csv")
df = df.drop(["name"], axis = 1)
train_set, test_set = train_test_split(df, random_state=42, test_size=0.2)
X_train = to_time_series_dataset(train_set.iloc[:, 1:])
X_test = to_time_series_dataset(test_set.iloc[:, 1:])
#y_train = train_set[:, 0].astype(np.int)
#y_test = test_set[:, 0].astype(np.int)

#X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
#X_train = X_train[y_train < 4] 
numpy.random.shuffle(X_train)

# Keep only 50 time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
# Make time series shorter
#X_train = TimeSeriesResampler(sz=15).fit_transform(X_train)
sz = X_train.shape[1]

euclidian_kmeans(X_train, seed, 20)
dba_kmeans(X_train, seed, 20)
#soft_dtw_kmeans(X_train, seed)

