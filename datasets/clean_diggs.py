import pandas as pd

def read_vote(fname):
    df = pd.read_csv(fname,delimiter=',',header=None)
    data = df.values.tolist()
    votes = {} # votes story:(vote user, time)


    for line in data:
        t, u, s = int(line[0]),str(line[1]),str(line[2])
        if s not in votes:
            votes[s] = {u: t}
        elif u not in votes[s]:
            votes[s][u] = t
        else:
            votes[s][u] = min(t, votes[s][u])
    return votes

def read_edges(fname):
    df = pd.read_csv(fname, delimiter=',', header=None)
    data = df.values.tolist()
    linklist = {}
    for line in data:
        mutual, t, s = int(line[0]),str(line[2]),str(line[3])
        if s not in linklist:
            linklist[s] = set()
        linklist[s].add(t)
        if mutual:
            if t not in linklist:
                linklist[t] = set()
            linklist[t].add(s)
    return linklist

def show_stats(list,name):
    n = len(list.keys())
    m = sum([len(each) for each in list.values()])
    d = 1.0*m/n
    print ("{} network\t n:{}\t m:{}\t d:{}".format(name,n,m,d))

def check_conditions(cascade,v,vt, edges):
    flag = 0
    #
    for u,t in cascade :
        if t<=vt and u in edges and v in edges[u]:
        # if t<=vt and v in edges[u] and u in mentions and v in mentions[u] and mentions[u][v] <= vt:
            flag += 1
    return (flag>0)

def gen_gt_cascade(retweets,edges):
    spreads = set()
    seeds = set()
    orders = {}
    node_step = {}
    for seed in retweets:
        # cascade=[each]

        x = sorted(retweets[seed].items(), key=lambda x: x[1])
        seed = x[0][0]
        cur_cascade= set([x[0]])
        ans = []
        candidates = set(x)
        while len(cur_cascade)>0:
            ans.append(cur_cascade)
            candidates = candidates-cur_cascade
            new_order_cascade = set()
            for each in candidates:
                v,t = each
                if check_conditions(cur_cascade,v,t,edges):
                    new_order_cascade.add(each)
            cur_cascade = new_order_cascade

        ans = ans[1:]

        if len(ans)>0:
            if seed in seeds:
                continue
            seeds.add(seed)
            v = []
            ord = len(ans)
            for i in range(len(ans)):
                step = i+1
                for each,_ in ans[i]:
                    if each not in node_step:
                        node_step[each] = step
                    node_step[each] = min(node_step[each],step)
            if ord not in orders:
                orders[ord] = 0
            orders[ord] += 1
            for i in range(len(ans)):
                v+=[u for u,_ in ans[i]]
            spreads.update(set(v))
    spreads = spreads.difference(seeds)
    distribution = sorted(orders.items(), key=lambda item: item[0])
    print ("new_retweets order distributions: ",distribution)
    invert_node_step = {}
    totalspread = len(spreads)
    for u in node_step:
        if u in seeds:
            continue
        step = node_step[u]
        if step not in invert_node_step:
            invert_node_step[step] = 0
        invert_node_step[step] += 1.0
    distribution = sorted(invert_node_step.items(), key=lambda item: item[0])
    for each in distribution:
        print(each)

    return seeds,spreads

def write_gt(edges,seeds,spreads):
    fo = open("digg.edgelist", "w")
    for u in edges:
        for v in edges[u]:
            fo.write("{} {}\n".format(u,v))
    fo = open("digg.seed", "w")
    for s in seeds:
        fo.write(s+"\n")
    fo = open("digg.spread", "w")
    for s in spreads:
        fo.write(s+"\n")


if __name__ == '__main__':
    votelist = read_vote("digg_votes.csv")
    show_stats(votelist, "vote")
    edgelist = read_edges("digg_friends.csv")
    show_stats(edgelist,"edges")
    seeds, spreads = gen_gt_cascade(votelist,edgelist)
    write_gt(edgelist,seeds,spreads)
