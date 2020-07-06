"""
Script used for analysing dataset using clustering.
"""
import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy
import sklearn.cluster
import sklearn.metrics
from plotly.offline import init_notebook_mode, plot
from sklearn import preprocessing


def show_optimal_n_for_k_means(path, norm=True, to_drop=['Country']):
    X = pd.read_csv('happiness_report_2017.csv')
    X = X.drop(to_drop, axis=1)
    if norm:
        X = normalize(X)
    clustering_scores = []
    for k in range(2, 11):
        clustering = sklearn.cluster.KMeans(n_clusters=k).fit(X)
        clustering_scores.append({
            'k': k,
            'sse': clustering.inertia_,
            'silhouette': sklearn.metrics.silhouette_score(X, clustering.labels_)
        })
    df_clustering_scores = pd.DataFrame.from_dict(clustering_scores, orient='columns')
    df_clustering_scores = df_clustering_scores.set_index('k')
    df_clustering_scores.sse.plot()
    plt.title('SSE score - Elbow')
    plt.show()
    plt.title('Silhouette index')
    df_clustering_scores.silhouette.plot()
    plt.show()


def show_optimal_n_for_agglomerative(path, linkage='single', to_drop=['Country']):
    X = pd.read_csv('happiness_report_2017.csv')
    X = X.drop(to_drop, axis=1)
    X = normalize(X)
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


def normalize(data_frame):
    x = data_frame.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=data_frame.columns)


def k_means(n, norm=True, to_drop=['Country']):
    X = pd.read_csv('happiness_report_2017.csv')
    X = X.drop(to_drop, axis=1)
    X = normalize(X)
    print(X.shape)
    clustering = sklearn.cluster.KMeans(n).fit(X)
    print(clustering.cluster_centers_)
    X = prep_data_to_show(clustering.labels_)
    # for i in range(n):
    #    print('Cluster {} has {} elements'.format(i,clustering.labels_.count(i)))
    X.to_csv("result-kmeans.csv")
    clusters = {}
    plotly_test(X)
    return clusters


def prep_data_to_show(labels):
    X = pd.read_csv('happiness_report_2017.csv')
    X['clusters'] = labels
    print(collections.Counter(labels))
    X = X.drop(
        ['Happiness.Rank', 'Happiness.Score', 'Whisker.high', 'Whisker.low', 'Economy..GDP.per.Capita.', 'Family',
         'Health..Life.Expectancy.',
         'Freedom', 'Generosity', 'Trust..Government.Corruption.',
         'Dystopia.Residual'], axis=1)
    return X


def aglomerative_clustering(n, linkage='single', to_drop=['Country']):
    X = pd.read_csv('happiness_report_2017.csv')
    X = X.drop(to_drop, axis=1)
    X = normalize(X)
    clustering = sklearn.cluster.AgglomerativeClustering(n, linkage=linkage).fit(X)
    clusters = {}
    X = prep_data_to_show(clustering.labels_)
    X.to_csv("result-aglo-{}.csv".format(linkage))
    plotly_test(X)
    return clusters


def show_distances_hist(index=0, to_drop=['Country']):
    X = pd.read_csv('happiness_report_2017.csv')
    X = X.drop(to_drop, axis=1)
    distance_matrix = scipy.spatial.distance_matrix(X, X)
    closest_neighbour_distance = np.where(distance_matrix == 0, 99, distance_matrix).min(axis=0)
    pd.Series(closest_neighbour_distance).hist(bins=40)
    plt.show()


def DBSCAN(eps, min_samples, index=0, to_drop=['Country']):
    X = pd.read_csv('happiness_report_2017.csv')
    X = X.drop(to_drop, axis=1)
    X = normalize(X)
    clustering = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    print(len(set(clustering.labels_)))
    X = prep_data_to_show(clustering.labels_)
    X.to_csv("result-DBSCAN.csv")
    plotly_test(X)
    # print(sklearn.metrics.silhouette_score(X,clustering.labels_))


def plotly_test(df_results):
    init_notebook_mode(connected=True)
    data = dict(type='choropleth',
                locations=df_results['Country'],
                locationmode='country names',
                z=df_results['clusters'],
                text=df_results['Country'],
                colorbar={'title': 'Cluster'})
    layout = dict(title='Happiness Index 2017 clustering',
                  geo=dict(showframe=False,
                           projection={'type': 'robinson'}))
    choromap3 = go.Figure(data=[data], layout=layout)
    plot(choromap3)


path = ''
to_drop = ['Country', 'Happiness.Rank', 'Happiness.Score', 'Whisker.high', 'Whisker.low', 'Dystopia.Residual']
# to_drop = ['Country','Happiness.Rank','Happiness.Score', 'Whisker.high', 'Whisker.low','Economy..GDP.per.Capita.', 'Health..Life.Expectancy.','Freedom', 'Generosity', 'Trust..Government.Corruption.','Dystopia.Residual']
# to_drop = ['Country', 'Happiness.Rank','Happiness.Score', 'Whisker.high', 'Whisker.low','Family', 'Health..Life.Expectancy.','Freedom','Dystopia.Residual']
# to_drop = ['Country', 'Happiness.Rank','Happiness.Score', 'Whisker.high', 'Whisker.low','Economy..GDP.per.Capita.','Generosity', 'Trust..Government.Corruption.','Dystopia.Residual']
k_means(3, True, to_drop)
aglomerative_clustering(3, 'complete', to_drop=to_drop)
# show_distances_hist(to_drop=to_drop)
# DBSCAN(0.25,3,to_drop=to_drop)
search_for_optimal_n = False
if search_for_optimal_n:
    d_index = 4
    show_optimal_n_for_k_means(path, to_drop=to_drop)
    # show_optimal_n_for_agglomerative(path,linkage='complete',to_drop=to_drop)
