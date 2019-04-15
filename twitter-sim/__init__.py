#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:33:57 2019

@author: dbeskow
"""

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import random
import numpy as np
from numpy.random import choice
from collections import Counter
from sklearn.metrics import pairwise_distances


#%%
def scale(x):
    x = np.array(x)
    return(x/np.max(x))
#%%
size = 20
prob = 0.3
influence_proportion = 0.1
bucket = [0,1]

#G = nx.erdos_renyi_graph(size, prob, seed=767, directed=True)
G = nx.scale_free_graph(size)
A = nx.adjacency_matrix(G).astype(bool)
similarity = 1 - pairwise_distances(A.todense(), metric = 'jaccard')
influence = scale(list(dict(G.degree()).values()))
#%%
def draw_simulation(graph, save = False):
    nodes = graph.nodes()
    colors = [graph.nodes[i]['belief'] for i in graph.nodes]
    pos = nx.spring_layout(graph, seed = 767)
    ec = nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colors, 
                                with_labels=False, node_size=100, cmap='YlOrRd')
    plt.colorbar(nc)
    plt.axis('off')
    plt.show()
#%%

for node, data in G.nodes(data=True):
    data['lambda'] = np.random.uniform(0.001,0.75)
    data['wake'] = 0 + np.round(np.random.exponential(scale = 1 / data['lambda']))
    data['inbox'] = []
    data['belief'] = np.random.uniform(0.0,1.0)
    data['susceptibility'] = np.random.uniform(0,1)
    
list(G.nodes(data=True))
draw_simulation(G)
#%%
total_tweets = []
all_beliefs = {'time':[],'user':[],'beliefs':[]}
global_perception = 0.0001
for step in range(3000):
    # Once a week we update the similarity matrix and Global Perception 
    if (step % 168) == 0:
        A = nx.adjacency_matrix(G).astype(bool)
        similarity = 1 - pairwise_distances(A.todense(), metric = 'jaccard')
        
        ## Update Global Perception
        if len(total_tweets) > 0:
            df = pd.concat(total_tweets)
            global_perception = df['tweets'].mean()
    for node, data in G.nodes(data=True):
        all_beliefs['time'].append(step)
        all_beliefs['user'].append(node);
        all_beliefs['beliefs'].append(data['belief']);
        if data['wake'] == step:
            # Get new 'wake' time
            data['wake'] = data['wake'] + np.round(np.random.exponential(scale = 1 / data['lambda']))
            # Read Tweets
            if len(data['inbox']) > 0:
                number_to_read = min(random.randint(0,20),len(data['inbox']))
                read_tweets = data['inbox'][-number_to_read:]
                perc = np.mean(read_tweets)
                #update influence
#                data['belief'] = data['belief'] + data['susceptibility'] * perc * (1-data['belief'])
                new_belief = data['belief'] +   (perc + global_perception) * (1-data['belief'])
                data['belief'] = new_belief
                
            # Send Tweets
            chance = data['belief'] * influence_proportion
            tweets = list(choice(bucket, np.random.randint(0,10),p=[1-chance, chance]))
            total_tweets.append(pd.DataFrame({'tweets': tweets, 'time' :[step] * len(tweets)}))
            predecessors = G.predecessors(node)
            for follower in predecessors:
                homophily = similarity[node,follower]
                importance = influence[follower]
                tweets = [homophily * importance * i for i in tweets]
                G.nodes[follower]['inbox'] = G.nodes[follower]['inbox'] + tweets
#list(G.nodes(data=True))
draw_simulation(G)


#%%
def draw_tweet_timeline(t_tweets, plot_type = 'area'):
    df = pd.concat(total_tweets)    
    df['type'] = 'noise'
    df.loc[df['tweets'] != 0,['type']] = 'disinformation'
    if plot_type == 'area':
        df.groupby(['time','type']).sum().unstack().plot.area()
    else:
        df.groupby(['time']).sum().plot(title = 'Tweets Per Hour')
draw_tweet_timeline(total_tweets)
#%%
def draw_tweet_bar(t_tweets):
    df = pd.concat(total_tweets)    
    df['type'] = 'noise'
    df.loc[df['tweets'] != 0,['type']] = 'disinformation'
    df = df['type'].value_counts()
    ax = df.plot(kind = 'bar', title = 'Tweets by Type',color = ['blue','red'], alpha = 0.6)
    for x,y in enumerate(df):
        ax.text(x,y,y)
draw_tweet_bar(total_tweets)
#%%
def draw_beliefs(all_beliefs, breaks = 'weeks'):
    df = pd.DataFrame(all_beliefs)  
    df2 = df[['time','beliefs']]
    if breaks == 'weeks':
        df2['time'] = df2['time']/168
        fig,ax = plt.subplots()
        df2 = df2.groupby(['time']).mean()
        df2.plot(title = 'Mean Belief', ax = ax)
#        df2.rolling(24).mean().plot(title = 'Mean Belief', ax = ax)
        ax.legend(["hourly avg", "daily rolling avg"])
        plt.ylabel('Belief Measure')
        plt.xlabel('time - weeks')
    else:
        fig,ax = plt.subplots()
        df2 = df2.groupby(['time']).mean()
        df2.plot(title = 'Mean Belief', ax = ax)
#        df2.rolling(24).mean().plot(title = 'Mean Belief', ax = ax)
        plt.ylabel('Belief Measure')
        plt.xlabel('time - hours')
#        ax.legend(["hourly avg", "daily rolling avg"])




draw_beliefs(all_beliefs)
#%%
from scipy.spatial import distance
distance.jaccard([0, 0], [0, 1, 0])    
    
nx.draw(G)
    
G.add_node(1, time='5pm')
G.nodes[0]['foo'] = 'bar'
G.nodes[0]['inbox'] = []
G.nodes[0]['inbox'].append('test')
list(G.nodes(data=True))

g = nx.Graph()
g.add_nodes_from([0, 1])

g.add_edge(0, 1)
nx.draw(g)


# Dataset
df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
 
# plot
df.plot.area()
