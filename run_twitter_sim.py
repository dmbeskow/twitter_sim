# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 22:56:27 2019
Description:
    This module allows users to run twitter_sim from the command line
@author: dmbeskow
"""

       
import pandas as pd
import networkx as nx
import random
import numpy as np
from numpy.random import choice
from sklearn.metrics import pairwise_distances
import progressbar
import argparse
import operator
#%%
def scale(x):
    '''
    Normalizes a vector by dividing each element by the vector max.
    '''
    x = np.array(x)
    return(x/np.max(x))
    
#%%
def link_prediction(G, node,similarity):
    '''
    This function takes the graph G, a given node, and the jaccard similarity 
    matrix for the nodes, and returns recommended link based on similarity.  
    '''
    ## Potential links are drawn from those whoe follow the same accounts 
    potential = []
    successors = G.successors(node)
    predecessors = list(G.predecessors(node)) 
    for successor in successors:
        friends = G.predecessors(successor)
        for friend in friends:
            if friend != node:
                potential.append(friend)
    # If potential exists, find highest similarity, otherwise sample from predecessors
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
    return(final)
 
#%%
def create_network(size = 100, bot_initial_links = 2, perc_bots = 0.05):
    '''
    Create basic single community scale free network.  Populate basic node data
    for normal users, bots, and stiflers.
    '''
    # Create Scale Free Graph
    G = nx.scale_free_graph(size)
    for node, data in G.nodes(data=True):
        data['lambda'] = np.random.uniform(0.001,0.75)
        data['wake'] = 0 + np.round(np.random.exponential(scale = 1 / data['lambda']))
        data['inbox'] = []
        data['mentioned_by'] = []
        data['belief'] = np.random.uniform(0,1.0)
        if data['belief'] < 0.2:
            data['kind'] = 'beacon'
        else:
            data['kind'] = 'normal'
        
    #  Add Bots
    num_bots = int(np.round(size*perc_bots))
    bot_names = [len(G) + i for i in range(num_bots)]
    for bot_name in bot_names:
        initial_links = random.sample(G.nodes, bot_initial_links)
        G.add_node(bot_name)
        for link in initial_links:
            G.add_edge(bot_name,link)
    # Add Bot Data      
    for node, data in G.nodes(data=True):
        if node in bot_names:
            data['lambda'] = np.random.uniform(0.1,0.75)
            data['wake'] = 0 + np.round(np.random.exponential(scale = 1 / data['lambda']))
            data['inbox'] = []
            data['belief'] = np.random.uniform(0.95,1.0)
            data['kind'] = 'bot'
            data['mentioned_by'] = []
        
    ## Remove self_loops and isololates
    G.remove_edges_from(list(G.selfloop_edges()))
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # Ensure every node has outdegree > 0 (otherwise similarity fails)
    A = nx.adjacency_matrix(G).astype(bool)
    b = np.squeeze(np.asarray(A.sum(axis = 1)))
    b = np.argwhere(b==0)
    for node in b:
        connected = [to for (fr, to) in G.edges(node)]
        unconnected = [n for n in G.nodes() if not n in connected] 
        new = random.sample(unconnected,1)
        G.add_edge(node[0], new[0])
        
    return(G)
    
#%%
def create_polarized_network(size = 100, bot_initial_links = 2, perc_bots = 0.05):
        '''
    Create polarized two community scale free network.  Populate basic node data
    for normal users, bots, and stiflers.
    '''
    # Create Scale Free Graph
    F = nx.scale_free_graph(size)
    H = nx.scale_free_graph(size)
    M = {}
    for num in range(size):
        M[num] = num + size
    H = nx.relabel_nodes(H, M, copy=False)
    G = nx.compose(F,H)
    
    for node, data in G.nodes(data=True):
        data['lambda'] = np.random.uniform(0.001,0.75)
        data['wake'] = 0 + np.round(np.random.exponential(scale = 1 / data['lambda']))
        data['inbox'] = []
        data['mentioned_by'] = []
        if node < size:
            data['belief'] = np.random.uniform(0,0.5)
        else:
            data['belief'] = np.random.uniform(0.5,1.0)
        if data['belief'] < 0.2:
            data['kind'] = 'beacon'
        else:
            data['kind'] = 'normal'
        
    #  Add Bots
    num_bots = int(np.round(len(G.nodes)*perc_bots))
    bot_names = [len(G) + i for i in range(num_bots)]
    for bot_name in bot_names:
        initial_links = random.sample(G.nodes, bot_initial_links)
        G.add_node(bot_name)
        for link in initial_links:
            G.add_edge(bot_name,link)
    # Add Bot Data      
    for node, data in G.nodes(data=True):
        if node in bot_names:
            data['lambda'] = np.random.uniform(0.1,0.75)
            data['wake'] = 0 + np.round(np.random.exponential(scale = 1 / data['lambda']))
            data['inbox'] = []
            data['belief'] = np.random.uniform(0.95,1.0)
            data['kind'] = 'bot'
            data['mentioned_by'] = []
        
    ## Remove self_loops and isololates
    G.remove_edges_from(list(G.selfloop_edges()))
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # Ensure every node has outdegree > 0 (otherwise similarity fails)
    A = nx.adjacency_matrix(G).astype(bool)
    b = np.squeeze(np.asarray(A.sum(axis = 1)))
    b = np.argwhere(b==0)
    for node in b:
        connected = [to for (fr, to) in G.edges(node)]
        unconnected = [n for n in G.nodes() if not n in connected] 
        new = random.sample(unconnected,1)
        G.add_edge(node[0], new[0])
        
    return(G)
#%%
def run(size = 100, perc_bots = 0.05, strategy = 'normal', polarized = 'normal'):
    '''
    This executes a single run of the twitter_sim ABM model
    '''
    ##################################
    influence_proportion = 0.01
    bucket1 = [0,1]
    bucket2 = [0,-1]
    probability_of_link = 0.05
    dynamic_network = True
    global_perception = 0.00000001
    retweet_perc = 0.25
    allowed_successors = 0.2
    
    # Create scale free network
    if polarized == 'polarized':
        G = create_polarized_network(size, perc_bots = perc_bots)
    else:
        G = create_network(size, perc_bots = perc_bots)
    
    #Create initial simlilarity and prestige arrays
    A = nx.adjacency_matrix(G).astype(bool)
    similarity = 1 - pairwise_distances(A.todense(), metric = 'jaccard')
    prestige = scale(list(dict(G.degree()).values()))
    
    # Initialize objects to collect results
    total_tweets = []
    all_beliefs = {'time':[],'user':[],'beliefs':[], 'kind':[]}
    bar = progressbar.ProgressBar()
    for step in bar(range(1680)):
        # Once a week we update the similarity matrix and Global Perception and prestige
        if (step % 168) == 0:
            A = nx.adjacency_matrix(G).astype(bool)
            similarity = 1 - pairwise_distances(A.todense(), metric = 'jaccard')
            prestige = scale(list(dict(G.in_degree()).values()))
            
            ## Update Global Perception
            if len(total_tweets) > 0:
                df = pd.concat(total_tweets)
                global_perception = 0.001*df['tweets'].mean()
        # Loop over all nodes
        for node, data in G.nodes(data=True):
            all_beliefs['time'].append(step)
            all_beliefs['user'].append(node);
            all_beliefs['beliefs'].append(data['belief']);
            all_beliefs['kind'].append(data['kind'])
            # Check if User logs on for this Time Step
            if data['wake'] < step:
                retweets = []
                # Get new 'wake' time
                data['wake'] = data['wake'] + np.round(np.random.exponential(scale = 1 / data['lambda']))
                # Read Tweets
                if len(data['inbox']) > 0:
                    number_to_read = min(random.randint(4,20),len(data['inbox']))
                    read_tweets = data['inbox'][-number_to_read:]
                    perc = np.mean(read_tweets)
                    # Update Belief
                    if (perc + global_perception) > 0:
                        new_belief = data['belief'] +   (perc + global_perception) * (1-data['belief'])
                    else:
                        new_belief = data['belief'] +   (perc + global_perception) * (data['belief'])
                    data['belief'] = new_belief  
                    # Get retweets from read tweets
                    retweets = random.sample(read_tweets, round(retweet_perc*len(read_tweets)))
                # Send Tweets for bots
                if data['kind'] == 'bot':
                    chance = 0.8
                    tweets = list(choice(bucket1, np.random.randint(0,10),p=[1-chance, chance]))
                    
                    # Send Tweets for Stiflers/Beacons
                elif (data['kind'] == 'beacon') and ('read_tweets' in locals()):
#                    chance = 0.8
    #                tweets = list(choice(bucket2, np.random.randint(0,10),p=[1-chance, chance]))
                    read_tweets = data['inbox'][-30:]
                    num_dis = np.sum(np.array(read_tweets) > 0)
                    tweets = [-1] * num_dis
                    
                # Send Tweets for normal users
                else:
                    chance = data['belief'] * influence_proportion
#                    chance = 0   # Normal users only send disinformation with retweets
                    tweets = list(choice(bucket1, np.random.randint(0,10),p=[1-chance, chance]))
                tweets.extend(retweets)
                total_tweets.append(pd.DataFrame({'tweets': tweets, 'time' :[step] * len(tweets)}))
                predecessors = G.predecessors(node)
                for follower in predecessors:
                    homophily = similarity[node,follower]
                    importance =  prestige[follower]
                    tweets = [homophily * importance * i for i in tweets]
                    G.nodes[follower]['inbox'].extend(tweets)
                    
                # Send Mentions
                neighbors = list(G.neighbors(node))
                mention = random.sample(neighbors,1)[0]
                G.nodes[mention]['mentioned_by'].append(node)
                    
                # Make sure doesn't have too many successors already
                successors = list(G.successors(node)) + [node]
                if len(successors) < allowed_successors * len(G.nodes) and (dynamic_network):
                    # If probabliliy right, add link for non-bot users
                    if (np.random.uniform(0,1) < probability_of_link) and (data['kind'] != 'bot'):
                        new_link = link_prediction(G,node,similarity)
                        if len(new_link) > 0:
                            G.add_edges_from(new_link) 

                    # If probabliliy right, add link to a mention
                    if (np.random.uniform(0,1) < probability_of_link) and (len(data['mentioned_by']) > 0):
                        new_link = random.sample(data['mentioned_by'],1)
                        if len(new_link) > 0:
                            G.add_edge(node, new_link[0]) 
                    # Bots try to add link every time
                    if (data['kind'] == 'bot'):
                        potential = list(set(G.nodes) - set(successors))
                        if len(potential) > 0:
                            if strategy == 'targeted':
                                degree = dict(G.in_degree(potential))
                                new_link = max(degree.items(), key=operator.itemgetter(1))[0]
                            else:
                                new_link = random.sample(list(potential),1)[0]
                            G.add_edge(node,new_link)
    return(pd.DataFrame(all_beliefs),pd.concat(total_tweets), G )
                

#%%

def main():
    
    parser=argparse.ArgumentParser(description="Simulations Project")
    parser.add_argument("-size",help="Number of nodes" , type=int, required=True)
    parser.add_argument("-perc",help="Percentage of bots" , type=float, required=True)
    parser.add_argument("-runs",help="Number of runs" ,type=int, required = True)
    parser.add_argument("-strategy",help="Bot Strategy" ,type=str, required = True)
    parser.add_argument("-polarized",help="Polarized" ,type=str, required = True)
    args=parser.parse_args()
    print(args)
    
    for i in range(args.runs):
        all_beliefs,total_tweets, G = run(size = args.size, perc_bots = args.perc, strategy = args.strategy, polarized = args.polarized)
        prefix = '_' + args.strategy +'_' + str(args.polarized) +  str(args.size) + '_' + str(round(args.perc,2)) + '_' + str(i) + '.csv'
        all_beliefs.to_csv('data/all_beliefs' + prefix, index = False)
        total_tweets.to_csv('data/total_tweets'+ prefix, index = False)

if __name__=="__main__":
    main()