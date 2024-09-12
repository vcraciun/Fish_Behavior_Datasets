import numpy
import matplotlib.pyplot as plt
import pandas as pd
from  sklearn.metrics import accuracy_score
from tslearn.clustering import TimeSeriesKMeans, silhouette_score, KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from sklearn.model_selection import train_test_split 
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import classification_report
import numpy as np

def euclidian_kmeans(X_train, seed, clusters):
    print("Euclidean k-means")
    sil_score_max = -1
    best_n_clusters = 0
    for n_clusters in range(2, clusters):
        km = TimeSeriesKMeans(n_clusters, verbose=False, random_state=seed, max_iter=100)                
        y_pred = km.fit_predict(X_train)
        sil_score = silhouette_score(X_train, y_pred)
        print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
    print(best_n_clusters)

    if best_n_clusters > 10:
        return

    km = TimeSeriesKMeans(best_n_clusters, verbose=False, random_state=seed, max_iter=100)        
    y_pred = km.fit_predict(X_train)
    print(km.labels_)
    plt.subplots(5,3, figsize =(10, 17))
    for yi in range(best_n_clusters):
        plt.subplot(5,3,yi+1)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "y-", alpha=.2)
        plt.plot(km.cluster_centers_[yi].ravel(), label = f"Cluster {yi+1}", color="r")
        if yi == 1:
            plt.title("Euclidean $k$-means", fontsize=20)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    plt.tight_layout()
    #plt.savefig(f"euclidian_kmeans_{best_n_clusters}.png", bbox_inches='tight', dpi=600)

def dba_kmeans(X_train, seed, clusters):
    print("DBA k-means")
    sil_score_max = -1
    best_n_clusters = 0
    for n_clusters in range(2, clusters):
        dba_km = TimeSeriesKMeans(n_clusters, n_init=2, metric="dtw", verbose=False, max_iter_barycenter=10, random_state=seed, max_iter=100)
        y_pred = dba_km.fit_predict(X_train)
        sil_score = silhouette_score(X_train, y_pred)
        print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
    print(best_n_clusters)

    if best_n_clusters > 10:
        return

    dba_km = TimeSeriesKMeans(best_n_clusters, n_init=2, metric="dtw", verbose=False, max_iter_barycenter=10, random_state=seed)
    y_pred = dba_km.fit_predict(X_train)
    print(dba_km.labels_)
    #plt.subplots(1,3,figsize =(15, 6))
    for yi in range(best_n_clusters):
        plt.subplot(5,3,yi+4)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "y-", alpha=.2)
        plt.plot(dba_km.cluster_centers_[yi].ravel(), label = f"Cluster {yi+1}", color="r")
        if yi == 1:
            plt.title("DBA $k$-means", fontsize=20)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    plt.tight_layout()
    #plt.savefig(f"DBA_kmeans_{best_n_clusters}.png", bbox_inches='tight', dpi=600)

def soft_dtw_kmeans(X_train, seed, specie, clusters):
    print("Soft-DTW k-means")
    sil_score_max = -1
    best_n_clusters = 0
    for n_clusters in range(2, clusters):
        sdtw_km = TimeSeriesKMeans(n_clusters, metric="softdtw", metric_params={"gamma": .01}, verbose=False, random_state=seed, max_iter=100)
        y_pred = sdtw_km.fit_predict(X_train)
        sil_score = silhouette_score(X_train, y_pred)
        print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
    print(best_n_clusters)

    if best_n_clusters > 10:
        return

    sdtw_km = TimeSeriesKMeans(best_n_clusters, metric="softdtw", metric_params={"gamma": .01}, verbose=False, random_state=seed)
    y_pred = sdtw_km.fit_predict(X_train)
    print(sdtw_km.labels_)
    #plt.subplots(1,3,figsize =(15, 6))
    for yi in range(best_n_clusters):
        plt.subplot(5,3,yi+7)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "y-", alpha=.2)
        plt.plot(sdtw_km.cluster_centers_[yi].ravel(), label = f"Cluster {yi+1}", color="r")
        if yi == 1:
            plt.title("Soft-DTW $k$-means", fontsize=20)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    plt.tight_layout()
    #plt.savefig(f"K-Means_{specie}_{best_n_clusters}.png", bbox_inches='tight', dpi=600)

def kshape_kmeans(X_train, seed, specie, clusters, y_test, y_train, X_test):
    print("k-Shape k-means")
    specs = {
        "zebra": "Zebrafish",
        "caras": "Goldfish"
    }
    sil_score_max = -1
    best_n_clusters = 0
    for n_clusters in range(2, clusters):
        ks = KShape(n_clusters, verbose=False, random_state=seed)
        y_pred = ks.fit_predict(X_train)
        sil_score = silhouette_score(X_train, y_pred)
        print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
    print(best_n_clusters)

    ks = KShape(best_n_clusters, verbose=False, random_state=seed)
    y_pred = ks.fit_predict(X_train)
    print(list(zip(y_train, y_pred)))

    toate = list(ks.labels_)
    unice = list(set(toate))
    freq = []
    for el in unice:
        freq += [list(ks.labels_).count(el)]
    print(freq)
    print("std: ", np.std(freq))

    rows = best_n_clusters // 2 
    cols = 2
    
    fig, ax = plt.subplots(rows,cols,figsize =((best_n_clusters//rows)*3,rows*3))
    for yi in range(best_n_clusters):
        plt.subplot(rows,cols,yi+1)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "y-", alpha=.2)
        plt.plot(ks.cluster_centers_[yi].ravel(), label = f"Cluster {yi+1} [{toate.count(yi)}]", color="r")
        plt.legend(fontsize=16)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    plt.suptitle(f"{specs[specie]} Train", fontsize=20)
    fig.supxlabel("Number of features (resampled to 40)", fontsize=20)
    fig.supylabel("Individual feature values", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"K-Means_train_{specie}.png", bbox_inches='tight', dpi=600)

    print("predictions ---------------------------------")
    y_pred = ks.predict(X_test)
    print(list(zip(y_test, y_pred)))

    toate = list(y_pred)

    fig, ax = plt.subplots(rows,cols,figsize =((best_n_clusters//rows)*3,rows*3))
    for yi in range(best_n_clusters):
        plt.subplot(rows,cols,yi+1)
        for xx in X_test[y_pred == yi]:
            plt.plot(xx.ravel(), "y-", alpha=.2)
        plt.plot(ks.cluster_centers_[yi].ravel(), label = f"Cluster {yi+1} [{toate.count(yi)}]", color="r")        
        plt.legend(fontsize=16)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    plt.suptitle(f"{specs[specie]} Test", fontsize=20)    
    fig.supxlabel("Number of features (resampled to 40)", fontsize=20)
    fig.supylabel("Individual feature values", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"K-Means_test_{specie}.png", bbox_inches='tight', dpi=600)

def plot_silhouette_score(specie): 
    silhouette = {
        "zebra": [
            [0.34,0.39,0.33,0.20,0.20,0.26,0.24,0.25,0.25,0.25,0.25,0.22,0.22,0.24,0.25,0.25,0.24,0.22,0.21,0.26,0.23,0.22,0.22,0.22,0.20,0.21,0.21,0.21],
            [0.37,0.40,0.32,0.33,0.32,0.27,0.27,0.28,0.29,0.27,0.28,0.28,0.28,0.28,0.28,0.27,0.24,0.24,0.23,0.24,0.23,0.22,0.23,0.23,0.23,0.22,0.23,0.21],
            [0.37,0.41,0.36,0.37,0.31,0.30,0.29,0.28,0.28,0.29,0.30,0.29,0.24,0.25,0.25,0.23,0.24,0.24,0.23,0.25,0.25,0.24,0.24,0.24,0.22,0.22,0.22,0.21],
            [0.17,0.12,0.07,-0.02,0.30,0.22,0.22,0.21,0.03,0.15,0.12,0.10,0.10,0.02,0.04,-0.03,0.05,0.04,0.02,0.07,0.01,0.00,-0.00,-0.02,-0.02,0.06,0.04,0.06],
            "Zebrafish"
        ],
        "caras": [
            [0.09,0.06,0.11,0.12,0.13,0.11,0.15,0.15,0.14,0.17,0.15,0.18,0.19,0.20,0.19,0.20,0.20,0.18,0.19,0.20,0.20,0.20,0.19,0.20,0.20,0.19,0.19,0.18],
            [0.18,0.22,0.22,0.19,0.22,0.21,0.22,0.23,0.24,0.22,0.22,0.23,0.23,0.25,0.20,0.22,0.20,0.25,0.22,0.22,0.23,0.23,0.25,0.26,0.22,0.21,0.24,0.21],
            [0.20,0.23,0.22,0.22,0.22,0.23,0.21,0.23,0.22,0.25,0.23,0.23,0.24,0.24,0.24,0.23,0.23,0.25,0.24,0.24,0.23,0.23,0.24,0.24,0.24,0.24,0.25,0.25],
            [0.13,0.12,0.08,0.08,0.09,0.13,0.16,0.07,0.09,0.04,0.13,0.10,0.12,-0.02,-0.01,0.06,0.01,0.15,0.06,0.03,0.04,0.09,0.01,0.09,0.07,0.13,0.14,0.10],
            "Goldfish"
        ]  
    }
    euc = silhouette[specie][0]
    dba = silhouette[specie][1]
    dtw = silhouette[specie][2]
    ksh = silhouette[specie][3]
    x_ax = list(range(2, 2+len(silhouette[specie][0])))
    plt.subplots(figsize =(8, 6)) 
    plt.plot(x_ax, euc, label = "Euclidean", linewidth=3)
    plt.plot(x_ax, dba, label = "DBA", linewidth=3)
    plt.plot(x_ax, dtw, label = "Soft-DTW", linewidth=3)
    plt.plot(x_ax, ksh, label = "k-Shape", linewidth=3)
    max_euc = max(euc)
    max_dba = max(dba)
    max_dtw = max(dtw)
    max_ksh = max(ksh)
    x_euc = euc.index(max_euc) + 2
    x_dba = dba.index(max_dba) + 2
    x_dtw = dtw.index(max_dtw) + 2
    x_ksh = ksh.index(max_ksh) + 2
    plt.plot([x_euc], [max_euc], "or", linewidth=3)
    plt.plot([x_dba], [max_dba], "or", linewidth=3)
    plt.plot([x_dtw], [max_dtw], "or", linewidth=3)
    plt.plot([x_ksh], [max_ksh], "or", linewidth=3)
    plt.legend(fontsize=20)
    plt.grid()
    plt.title(f"K-Means silhouette analysis : {silhouette[specie][4]}", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Number of possible clusters", fontsize=20)
    plt.ylabel("Silhouette value / cluster count", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"K-Means_{specie}_silhouette.png", bbox_inches='tight', dpi=600)

def process_specie(specie, clusters):
    seed = 0
    numpy.random.seed(seed)
    df = pd.read_csv(f"{specie}_all_features.csv")
    df = df.drop(["name"], axis = 1)
    train_set, test_set = train_test_split(df, random_state=42, test_size=0.2)
    X_train = to_time_series_dataset(train_set.iloc[:, 1:])
    X_test = to_time_series_dataset(test_set.iloc[:, 1:])
    y_train = train_set.iloc[:, 0]
    y_test = test_set.iloc[:, 0]
    numpy.random.shuffle(X_train)
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
    X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)

    X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
    X_test = TimeSeriesResampler(sz=40).fit_transform(X_test)

    plot_silhouette_score(specie)

    #euclidian_kmeans(X_train, seed, clusters)
    #dba_kmeans(X_train, seed, clusters)
    #soft_dtw_kmeans(X_train, seed, specie, clusters)    
    
    kshape_kmeans(X_train, seed, specie, clusters, y_test, y_train, X_test)

process_specie("zebra", 8)
process_specie("caras", 10)
