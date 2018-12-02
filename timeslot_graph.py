from tqdm import tqdm
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def add_timeslot_weight_to_graph(dataset, graph, slot_length=2):
    def add_od_to_graph_with_timeslot(graph, o, d, slot_index, prefix='weight_'):
        try:
            weight = graph[o][d]['weight']
            graph[o][d]['weight'][slot_index] += 1
        except KeyError:
            graph.add_nodes_from([o, d])
            graph.add_edge(o, d)
            graph[o][d]['weight'] = [0] * int(24 / slot_length)
            
    with tqdm(total=int(dataset.shape[0] / 2), desc='Adding dataset to graph: ') as pbar:
        for orderid, group in dataset.groupby('orderid'):
            slot_index = int(group.iloc[0]['timestamp'].hour / slot_length)
            add_od_to_graph_with_timeslot(graph, group.iloc[0]['cluster'], group.iloc[1]['cluster'], 
                                         slot_index, prefix='slot_')
            pbar.update(1)


def draw_timeslot_graph(graph, cluster_centers, od_set, slot_range,\
                        point_color='red', point_style='D', arrow_color='#f9f568',\
                        figsize=(10, 10), dpi=100, thre=50, norm=100, save_file=None):
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
        
    for edge in graph.edges:
        start, end = edge[0:2]
        if start == end:
            continue
        
        weight = sum(graph[start][end]['weight'][slot_range[0]:(slot_range[1]+1)])
        if weight < thre:
            continue

        x1 = cluster_centers[start][0]
        y1 = cluster_centers[start][1]
        x2 = cluster_centers[end][0]
        y2 = cluster_centers[end][1]
        width = (weight-thre)/norm

        ax.annotate('', xy = (x2, y2),xytext = (x1, y1), fontsize = 7, color=arrow_color, 
                    arrowprops=dict(edgecolor='black', facecolor=arrow_color, shrinkA=0, 
                                    shrinkB=0, width=width, headwidth=max(width*2, 4), alpha=0.8))
    
    plt.scatter(od_set['longitude'], od_set['latitude'], alpha=0.1, s=1, c='#3677af')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=point_color, marker=point_style, s=40)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
