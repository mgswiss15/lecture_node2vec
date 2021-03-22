''' Magda Gregorova 21/3/2021 - helpers for lecture on node2vec'''

import pandas as pd
import networkx as nx
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)
import math
from node2vec import Node2Vec
import numpy as np
import scipy
import random
from sklearn.cluster import KMeans


def load_got(path):

    # load data from path or disk
    try:
        got_data = pd.read_csv(path)
    except:
        print(f"Cannot load from {path}.")
        print(f"Using local asoiaf-book1-edges.csv file instead.")
        got_data = pd.read_csv("asoiaf-book1-edges.csv")

    # some cleaning
    got_data.drop(['Type', 'book'], axis=1, inplace=True)

    # create nx network
    got_net = nx.Graph()

    for _, row in got_data.iterrows():
        got_net.add_node(row['Source'])
        got_net.add_node(row['Target'])
        # got_net.add_edge(row['Source'], row['Target'])
        got_net.add_edge(row['Source'], row['Target'], weight=row['weight'])

    # get characters with degrees
    chars = pd.DataFrame(data=sorted(got_net.degree, key=lambda char: char[1], reverse=True), columns=['id', 'degree'])

    got = {'data': got_data, 'net': got_net, 'chars': chars}
    return got

def plot_net(G, houses=False, degrees=False, weights=False, walks=False, labels=False, gsize=800):

    # get walks
    pairs = []
    starts = []
    ends = []
    if walks:
        Walks = get_walks(G, num_walks=walks['num'], p=walks.get('p', 1), q=walks.get('q', 1))
        print("Walks")
        for walk in Walks:
            print(walk)
            for i in range(len(walk)-1):
                pairs.append((walk[i], walk[i+1]))
                pairs.append((walk[i+1], walk[i]))
            starts.append(walk[0])
            ends.append(walk[-1])

    # choose the layout we want
    pos = nx.nx_agraph.graphviz_layout(G, prog='neato')

    # nodes    
    Xn = [pos[k][0] for k in G.nodes]
    Yn = [pos[k][1] for k in G.nodes]
    textn = list(G.nodes)
    widthline=1

    if degrees:
        Colorn = [x[1] for x in G.degree]
        Sizen = [(math.log(x+1))*5 for x in Colorn]
        textn = [f"{x}: {y}" for x, y in zip(G.nodes, Colorn)]
        if houses:
            Colorn = get_houses(G)
    elif houses:
        Colorn = get_houses(G)
        Sizen = 8
    else:
        Colorn = 'blue'
        Sizen = 8
    if type(labels) is np.ndarray:
        Colorn = labels.tolist()
        textn = [f"{x}: {y}" for x, y in zip(G.nodes, Colorn)]
    Colorline = 'black'
    if walks:
        Colorline = []
        widthline = []
        for i in range(len(Xn)):
            c = 'blue'
            w = 1
            if list(G.nodes)[i] in starts:
                c = 'lime'
                w = 3
            if list(G.nodes)[i] in ends:
                c = 'red'
                w = 3
            Colorline.append(c)
            widthline.append(w)

    trace_nodes = dict(type='scatter', x=Xn, y=Yn, mode='markers', marker=dict(size=Sizen, color=Colorn, line=dict(width=widthline, color=Colorline)),
                       text=textn, hoverinfo='text')

    # edges - each in a new graph
    edge_traces = []
    for e in G.edges():
        weight = 0.15
        color = 'blue'
        if weights:
            weight = G.get_edge_data(*e)['weight']/20
        if e in pairs:
            color='purple'
            weight = 3
        et = make_edge([pos[e[0]][0], pos[e[1]][0], None], [pos[e[0]][1], pos[e[1]][1], None], weight, color)
        edge_traces.append(et)

    # graph
    # hide axis line, grid, ticklabels
    axis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')

    titleG = f"Game of Thrones: {G.order()} characters, {G.size()} edges."
    layout=dict(title=titleG, font= dict(family='Arial'),
                height=gsize, width=gsize, autosize=False, showlegend=False, xaxis=axis, yaxis=axis,
                margin=dict(l=0.05*gsize, r=0.05*gsize, b=0.05*gsize, t=0.1*gsize, pad=0,), hovermode='closest',
                plot_bgcolor='white')

    fig = dict(data=[*edge_traces, trace_nodes], layout=layout)
    return iplot(fig)

def make_edge(x, y, width, color):
    """
    Args:
        x: a tuple of the x from and to, in the form: tuple([x0, x1, None])
        y: a tuple of the y from and to, in the form: tuple([y0, y1, None])
        width: The width of the line

    Returns:
        a Scatter plot which represents a line between the two points given. 
    """
    return dict(type='scatter', mode='lines',
                x=x,
                y=y,
                line=dict(width=width,color=color),
                hoverinfo='none')

def get_houses(G):
    #make the colour map that we can use to colour code the nodes in our graph
    color_map = []
    for i in G.nodes:
        if house(i,'Lannister'):
            color_map.append('blue')
        elif house(i,'Targaryen'):   
            color_map.append('red')
        elif house(i,'Baratheon'):   
            color_map.append('green')
        elif house(i,'Tully'):   
            color_map.append('orange')
        elif house(i,'Frey'):   
            color_map.append('purple')
        elif house(i,'Stark'):   
            color_map.append('lightblue')
        elif house(i,'Jon-Snow'):   
            color_map.append('lightblue')
        elif house(i,'Royce'):   
            color_map.append('lime')
        elif house(i,'Tarly'):   
            color_map.append('yellow')
        elif house(i,'Tyrell'):   
            color_map.append('teal')
        elif house(i,'Arryn'):   
            color_map.append('brown')
        elif house(i,'Greyjoy'):   
            color_map.append('deeppink')
        elif house(i,'Cassel'):   
            color_map.append('chocolate')
        elif house(i,'Bolton'):   
            color_map.append('maroon')
        elif house(i,'Mormont'):   
            color_map.append('blueviolet')
        elif house(i,'Pool'):   
            color_map.append('olive')
        elif house(i,'Karstark'):   
            color_map.append('coral')
        elif house(i,'Clegane'):   
            color_map.append('crimson')
        elif house(i,'Piper'):   
            color_map.append('cornflowerblue')
        else: 
            color_map.append('dimgrey')       

    return color_map
    
def house(x,house):
    import regex as re
    if re.search(f'^(?!.*details\.cfm).*{house}.*$',x):
        return True
    else:
        return False

def get_subgraph(G, chars):
    nbunch = []
    for char in chars:
        nbunch.extend(list(G.neighbors(char)))
    nbunch.extend(chars)
    return G.subgraph(nbunch)


def get_walks(G, walk_length=5, num_walks=5, p=1, q=1):
    node2vec = Node2Vec(G, dimensions=16, walk_length=walk_length, num_walks=1, workers=1, quiet=True, p=p, q=q)  # Use temp_folder for big graphs
    walks = node2vec.walks
    degrees = [G.degree[walk[0]] for walk in walks]
    degrees = np.asarray(degrees)
    probs = degrees/degrees.sum()
    # idx = np.random.choice(range(len(walks)), size=num_walks, p=probs, replace=False)
    idx = np.random.choice(range(len(walks)), size=num_walks, replace=False)
    sample = [walks[i] for i in idx]
    return sample

def train_embeddings(G, p=1, q=1):
    node2vec = Node2Vec(G, dimensions=16, walk_length=10, num_walks=100, workers=1, p=p, q=q, quiet=True)  # Use temp_folder for big graphs
    model = node2vec.fit(window=6, min_count=1, batch_words=5)
    fname = f"wv_{p}_{q}.w2v"
    # model.wv.save_word2vec_format(fname)
    return model.wv


def embeddings(G, names, p, q, c, gsize=800):
    G = get_subgraph(G, names)
    wv = train_embeddings(G, p, q)
    wordvecs = wv[list(G.nodes)]
    kmeans = KMeans(n_clusters=c, random_state=0).fit(wordvecs)
    print("Examples of embeddings")
    for i in range(3):
        print(f"{list(G.nodes)[i]}: {wordvecs[i]}")
    plot_net(G, labels=kmeans.labels_, degrees=True, gsize=gsize)


def plot_walks(G, names, p, q, num_walks=3):
    G = get_subgraph(G, names)
    plot_net(G, degrees=True, houses=True, walks={'num': num_walks, 'p': p, 'q': q}, gsize=600)

