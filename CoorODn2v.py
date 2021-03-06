import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from tqdm import *
import matplotlib.pyplot as plt


class CoorODn2v:
    '''
    Coordinate node2vec model.
    Using KMeans cluster algorithm to cluster sparse points with coordinates,
    using originate-destination point pairs to generate a weighted graph,
    finally using node2vec to generate embeddings for each point.
    '''
    
    def __init__(self, clusters=50, dimensions=64):
        '''
        :param clusters: Number of clusters in KMeans model.
        :param dimensions: Length of embedding vector.
        '''
        self.clusters = clusters
        self.dimensions = dimensions
        
        self.od_set = None
        self.kmeans = None
        self.n2v_wv = None
        self.graph = None
    
    def fit(self, dataset, walk_length=10, num_walks=200, weight_key='weight', workers=47):
        '''
        :param dataset: Coordinate dataset. Must at least contains coordinate columns `['longitude', 'latitude']`,
        time column `['timestamp']` and group index `['orderid']`.
        '''
        def fetch_od(dataset: pd.DataFrame):
            order_series = dataset['orderid'].drop_duplicates()

            result = []
            with tqdm(total=order_series.shape[0], desc='Fetching O-D point pairs: ') as tqdm_iter:
                for orderid, group in dataset.groupby('orderid'):
                    group = group.sort_values('timestamp')
                    result.append(group.iloc[0])
                    result.append(group.iloc[-1])
                    tqdm_iter.update(1)
            result = pd.DataFrame(result)
            result.index = range(result.shape[0])

            return result
        
        def add_od_to_graph(graph: nx.Graph, o, d):
            try:
                graph[o][d]['weight'] += 1
            except KeyError:
                graph.add_nodes_from([o, d])
                graph.add_edge(o, d, weight=1)
                
        def add_dataset_to_graph(dataset: pd.DataFrame, graph):
            with tqdm(total=dataset.shape[0] / 2, desc='Adding dataset to graph: ') as tqdm_iter:
                for orderid, group in dataset.groupby('orderid'):
                    add_od_to_graph(graph, group.iloc[0]['cluster'], group.iloc[1]['cluster'])
                    tqdm_iter.update(1)
        
        self.graph = nx.DiGraph()
        
        self.od_set = fetch_od(dataset)
        self.od_set.index = range(self.od_set.shape[0])
        
        print('Fitting KMeans model...')
        self.kmeans = KMeans(n_clusters=self.clusters, n_jobs=workers).fit(self.od_set[['longitude', 'latitude']])
        self.od_set['cluster'] = self.kmeans.labels_
        
        add_dataset_to_graph(self.od_set, self.graph)
        
        n2v = Node2Vec(self.graph, dimensions=self.dimensions, walk_length=10, num_walks=200, 
                       workers=workers, weight_key='weight')
        print('Processing random walk...')
        model = n2v.fit(window=10, min_count=1, batch_words=4)
        
        self.n2v_wv = model.wv
        print('Finished fitting model!')
        
    def predict(self, test_set):
        if self.kmeans is None:
            raise ValueError('Model is not trained yet.')
            
        result = pd.DataFrame(test_set, copy=True)
        kmeans_labels = self.kmeans.predict(result[['longitude', 'latitude']])
        result['cluster'] = kmeans_labels
        
        embeddings = []
        with tqdm(total=result.shape[0], desc='Predicting coordinate embeddings: ') as pbar:
            for index, row in result.iterrows():
                embed_row = self.n2v_wv.word_vec(str(row['cluster']))
                embeddings.append(embed_row)
                pbar.update(1)
        
        embeddings = pd.DataFrame(embeddings, columns=['embed_%d' % i for i in range(self.dimensions)])
        
        return embeddings
    
    def plot_clusters(self, figsize=(10, 10), dpi=100, draw_arrows=False, thre=200, norm=300, color='#f9f568'):
        if self.od_set is None:
            raise ValueError('Model is not trained yet.')
        
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        
        if draw_arrows:
            self.draw_graph_arrows(ax, thre=thre, norm=norm, color=color)
        
        plt.scatter(self.od_set['longitude'], self.od_set['latitude'], alpha=0.2, s=1, c='#3677af')
        centers = self.kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='D', s=40)
        plt.show()
        
    def draw_graph_arrows(self, ax, color='red', thre=50, norm=10):
        coor = self.kmeans.cluster_centers_
        
        for edge in self.graph.edges:
            start, end = edge[0:2]
            if start == end:
                continue
            weight = self.graph[start][end]['weight']

            if weight < thre:
                continue

            x1 = coor[start][0]
            y1 = coor[start][1]
            x2 = coor[end][0]
            y2 = coor[end][1]
            width = (weight-thre)/norm

            ax.annotate('', xy = (x2, y2),xytext = (x1, y1), fontsize = 7, color=color, 
                        arrowprops=dict(edgecolor='black', facecolor=color, shrinkA=0, 
                                        shrinkB=0, width=width, headwidth=max(width*3, 4)))