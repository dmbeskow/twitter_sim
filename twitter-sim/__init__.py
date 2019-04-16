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
import progressbar

#%%
def scale(x):
    x = np.array(x)
    return(x/np.max(x))
    
#%%
def link_prediction(G, node):
    potential = []
    successors = G.successors(node)
    predecessors = list(G.predecessors(node)) 
    for successor in successors:
        friends = G.predecessors(successor)
        for friend in friends:
            if friend != node:
                potential.append(friend)
    final = []
    if len(potential) > 0:
        jaccard1 = similarity[node,potential]
        i = np.argmax(jaccard1)
        link = (node,potential[i])
        if ~G.has_edge(link[0],link[1]):
            final.append(link)
    elif len(predecessors) > 0:
        get_one = random.sample(list(predecessors),1)
        link = (node,get_one[0])
        if ~G.has_edge(link[0],link[1]):
            final.append(link)
#    else:
#        all_nodes = list(G.nodes)
#        all_nodes.remove(node)
#        get_one = random.sample(all_nodes,1)
#        final.append((node,get_one[0]))
    return(final)
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
size = 200
prob = 0.3
influence_proportion = 0.1
bucket1 = [0,1]
bucket2 = [0,-1]
probability_of_link = 0.1
dynamic_network = True
global_perception = 0.001
similarity_weight = 3
prestige_weight = 1

#G = nx.erdos_renyi_graph(size, prob, seed=767, directed=True)
G = nx.scale_free_graph(size)

## Remove self_loops and isololates
G.remove_edges_from(list(G.selfloop_edges()))
G.remove_nodes_from(list(nx.isolates(G)))

A = nx.adjacency_matrix(G).astype(bool)
b = np.squeeze(np.asarray(A.sum(axis = 1)))
b = np.argwhere(b==0)
for node in b:
    connected = [to for (fr, to) in G.edges(node)]
    unconnected = [n for n in G.nodes() if not n in connected] 
    new = random.sample(unconnected,1)
    G.add_edge(node[0], new[0])

A = nx.adjacency_matrix(G).astype(bool)
similarity = 1 - pairwise_distances(A.todense(), metric = 'jaccard')
prestige = scale(list(dict(G.degree()).values()))
#%%

for node, data in G.nodes(data=True):
    data['lambda'] = np.random.uniform(0.001,0.75)
    data['wake'] = 0 + np.round(np.random.exponential(scale = 1 / data['lambda']))
    data['inbox'] = []
    data['belief'] = np.random.uniform(0.0,1.0)
    data['kind'] = 'normal'
#    data['susceptibility'] = np.random.uniform(0,1)
    
#list(G.nodes(data=True))
draw_simulation(G)

#list(G.selfloop_edges())
#%%
total_tweets = []
all_beliefs = {'time':[],'user':[],'beliefs':[]}
bar = progressbar.ProgressBar()
for step in bar(range(2000)):
    # Once a week we update the similarity matrix and Global Perception and prestige
    if (step % 168) == 0:
        A = nx.adjacency_matrix(G).astype(bool)
        similarity = 1 - pairwise_distances(A.todense(), metric = 'jaccard')
        prestige = scale(list(dict(G.degree()).values()))
        
        ## Update Global Perception
        if len(total_tweets) > 0:
            df = pd.concat(total_tweets)
            global_perception = 0.001*df['tweets'].mean()
    for node, data in G.nodes(data=True):
        all_beliefs['time'].append(step)
        all_beliefs['user'].append(node);
        all_beliefs['beliefs'].append(data['belief']);
        if data['wake'] < step:
            # Get new 'wake' time
            data['wake'] = data['wake'] + np.round(np.random.exponential(scale = 1 / data['lambda']))
            # Read Tweets
            if len(data['inbox']) > 0:
                number_to_read = min(random.randint(0,20),len(data['inbox']))
                read_tweets = data['inbox'][-number_to_read:]
                perc = np.mean(read_tweets)
                #update prestige
#                data['belief'] = data['belief'] + data['susceptibility'] * perc * (1-data['belief'])
                if perc > 0:
                    new_belief = data['belief'] +   (perc + global_perception) * (1-data['belief'])
                else:
                    new_belief = data['belief'] +   (perc + global_perception) * (data['belief'])
                data['belief'] = new_belief     
            # Send Tweets for bots
            if data['kind'] == 'bot':
                chance = 0.8
                tweets = list(choice(bucket1, np.random.randint(0,10),p=[1-chance, chance]))
            # Send Tweets for normal users
            elif data['belief'] > 0.05:
                chance = data['belief'] * influence_proportion
                tweets = list(choice(bucket1, np.random.randint(0,10),p=[1-chance, chance]))
            # Send Tweets for Beacons
            else:
                chance = 0.8
                tweets = list(choice(bucket2, np.random.randint(0,10),p=[1-chance, chance]))
            total_tweets.append(pd.DataFrame({'tweets': tweets, 'time' :[step] * len(tweets)}))
            predecessors = G.predecessors(node)
            for follower in predecessors:
                homophily = similarity_weight * similarity[node,follower]
                importance = prestige_weight * prestige[follower]
                tweets = [homophily * importance * i for i in tweets]
                G.nodes[follower]['inbox'] = G.nodes[follower]['inbox'] + tweets
                
            # If probabliliy right, add link
            if (np.random.uniform(0,1) < probability_of_link) and (dynamic_network):
                new_link = link_prediction(G,node)
                if len(new_link) > 0:
                    G.add_edges_from(new_link)                
#x = list(G.nodes(data=True))
draw_simulation(G)
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
from scipy.spatial import distance
distance.jaccard([0, 0], [0, 1, 0])    
    
nx.draw(G2, with_labels = True)
    
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


G = nx.complete_graph(5)
G2 = nx.Graph(G.to_undirected())
preds = nx.adamic_adar_index(G2.to_undirected())
for u, v, p in preds:
    print('(%d, %d) -> %.8f' % (u, v, p))
