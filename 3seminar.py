import numpy as np
import sklearn.cluster, sklearn.metrics
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

paths = ['densegrid.csv', 'clusters5.csv','clusters5n.csv','boxes.csv','annulus.csv']

def test_func():
    for path in paths:
        X = np.loadtxt(path,delimiter=';')
        #clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=5, affinity='euclid', linkage='complete').fit(X)
        #plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_)
        #plt.show()

        clustering = sklearn.cluster.DBSCAN(eps=0.08,min_samples=5).fit(X)
        plt.scatter(X[:,0], X[:,1], c=clustering.labels_)
        plt.show()

def show_distances():
    X = np.loadtxt(paths[0], delimiter=';')
    neigh = NearestNeighbors(n_neighbors=3)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()

def show_distances_hist(index=0):
    X = np.loadtxt(paths[index], delimiter=';')
    distance_matrix = scipy.spatial.distance_matrix(X, X)
    closest_neighbour_distance = np.where(distance_matrix==0, 99, distance_matrix).min(axis=0)
    pd.Series(closest_neighbour_distance).hist(bins=40)
    plt.show()


def show_optimal_n_for_k_means(path):
    X = np.loadtxt(path,delimiter=';')
    clustering_scores = []
    for k in range(2, 11):
        clustering = sklearn.cluster.KMeans(n_clusters=k).fit(X)
        clustering_scores.append({
            'k': k,
            'sse': clustering.inertia_,
            'silhouette': sklearn.metrics.silhouette_score(X, clustering.labels_),
            'chs': sklearn.metrics.calinski_harabasz_score(X, clustering.labels_)
        })
    df_clustering_scores = pd.DataFrame.from_dict(clustering_scores, orient='columns')
    df_clustering_scores = df_clustering_scores.set_index('k')
    df_clustering_scores.sse.plot()
    plt.show()
    df_clustering_scores.silhouette.plot()
    plt.show()
    df_clustering_scores.chs.plot()
    plt.show()

def show_optimal_n_for_agglomerative(path, linkage='single'):
    X = np.loadtxt(path,delimiter=';')
    clustering_scores = []
    for k in range(2, 11):
        clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(X)
        clustering_scores.append({
            'k': k,
            'silhouette': sklearn.metrics.silhouette_score(X, clustering.labels_)
        })
    df_clustering_scores = pd.DataFrame.from_dict(clustering_scores, orient='columns')
    df_clustering_scores = df_clustering_scores.set_index('k')
    df_clustering_scores.silhouette.plot()
    plt.show()



def k_means(n):
    for path in paths:
        X = np.loadtxt(path,delimiter=';')
        #clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=5, affinity='euclid', linkage='complete').fit(X)
        #plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_)
        #plt.show()

        clustering = sklearn.cluster.KMeans(n).fit(X)
        plt.scatter(X[:,0], X[:,1], c=clustering.labels_)
        plt.show()

def aglomerative_clustering(n, linkage='single',index=0):
    #for path in paths:
    X = np.loadtxt(paths[index],delimiter=';')
    #clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=5, affinity='euclid', linkage='complete').fit(X)
    #plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_)
    #plt.show()

    clustering = sklearn.cluster.AgglomerativeClustering(n, linkage=linkage).fit(X)
    plt.scatter(X[:,0], X[:,1], c=clustering.labels_)
    plt.show()
    
def DBSCAN(eps,min_samples,index=0):
    #for path in paths[:1]:
    X = np.loadtxt(paths[index],delimiter=';')
    #clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=5, affinity='euclid', linkage='complete').fit(X)
    #plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_)
    #plt.show()
    clustering = sklearn.cluster.DBSCAN(eps=eps,min_samples=min_samples).fit(X)
    print(len(set(clustering.labels_)))
    print(sklearn.metrics.silhouette_score(X,clustering.labels_))
    plt.scatter(X[:,0], X[:,1], c=clustering.labels_)
    plt.show()


def DBSCAN_search():
    #for path in paths[:1]:
    X = np.loadtxt(path,delimiter=';')
    #clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=5, affinity='euclid', linkage='complete').fit(X)
    #plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_)
    #plt.show()
    eps_list = np.arange(0.001,1,0.001)
    print(len(eps_list))
    min_samples_list = range(1,10)
    for eps in eps_list:
        for min_samples in min_samples_list:
            clustering = sklearn.cluster.DBSCAN(eps=eps,min_samples=min_samples).fit(X)
            if len(set(clustering.labels_)) < 5 and len(set(clustering.labels_)) > 2:
                print("len:{}, eps:{}, smaples:{}".format(len(set(clustering.labels_)),eps,min_samples))
    #plt.scatter(X[:,0], X[:,1], c=clustering.labels_)
    #plt.show()

def draw_set(index):
    X = np.loadtxt(paths[index],delimiter=';')
    plt.scatter(X[:,0], X[:,1])
    plt.show()



#test_func()
#show_distances()

#show_distances_hist(1)

#aglomerative_clustering(2)
#DBSCAN(1.5,3,1)
DBSCAN(0.19,9,0)
#DBSCAN(0.18,11,0)
DBSCAN(1.7,2,4)
#DBSCAN(0.6,3,1)
DBSCAN(2.0,8,2)

#DBSCAN_search()
#for path in paths:
#    show_optimal_n_for_k_means(path)
#aglomerative_clustering(2,index=4,linkage='ward')
if True:
    d_index=4
    draw_set(d_index)
    show_optimal_n_for_k_means(paths[d_index])
    show_optimal_n_for_agglomerative(paths[d_index],linkage='single')