import networkx as nx

def load_graph(fname):
    fo = open(fname, "r")
    linklist = []

    for line in fo.readlines():
        line = line.rstrip("\r\n")
        u,v = line.split(" ")
        linklist.append(tuple([u,v]))
    G = nx.Graph()
    G.add_edges_from(linklist)
    # add meeting probability
    meeting = {each: 5.0 / (5 + G.degree[each]) for each in G.nodes()}
    weight = {each: 1.0 / G.degree[each] for each in G.nodes()}
    for each in G.nodes():
        for u,v in G.edges(each):
            G[u][v]['weight'] = weight[v]
            G[u][v]['meet'] = meeting[u]
    return G


def load_seeds(fname):
    seeds = []
    fo = open(fname, "r")
    for line in fo.readlines():
        s = line.rstrip("\r\n")
        seeds.append(s)
    # print("seeds top5: " + str(seeds[:5]))
    return seeds
