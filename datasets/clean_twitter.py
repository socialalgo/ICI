
def add_entry(dicts, key, val_key, val_val):
    if key not in dicts:
        dicts[key] = {val_key:val_val}
    else:
        dicts[key][val_key] = val_val

def read_activity(fname):
    fo = open(fname, "r")
    retweet = {} # seed tweet:(retweet user, retweet time)
    mention = {} # who: (mentioned whom, mention time)
    reply = {}  # who: (replied whom, reply time)

    for line in fo.readlines():
        line = line.rstrip("\r\n")
        u, v, t, flag = line.split(" ")
        t = int(t)
        if flag == 'MT':
            if u not in mention:
                mention[u] = {v: t}
            elif v not in mention[u]:
                mention[u][v] = t
            else:
                mention[u][v]=min(t,mention[u][v])
        elif flag == 'RT':
            if v not in retweet:
                retweet[v] = {u: t}
            elif u not in retweet[v]:
                retweet[v][u] = t
            else:
                retweet[v][u]=min(retweet[v][u],t)
        elif flag == 'RE':
            if u not in reply:
                reply[u] = {v: t}
            elif v not in reply[u]:
                reply[u][v] = t
            else:
                reply[u][v]=min(reply[u][v],t)
    return mention,reply,retweet

def read_edges(fname):
    fo = open(fname, "r")
    linklist = {}
    for line in fo.readlines():
        line = line.rstrip("\r\n")
        u,v = line.split(" ")
        if u not in linklist:
            linklist[u] = set()
        linklist[u].add(v)
    return linklist

def show_stats(list,name):
    n = len(list.keys())
    m = sum([len(each) for each in list.values()])
    d = 1.0*m/n
    print ("{} network\t n:{}\t m:{}\t d:{}".format(name,n,m,d))

def check_conditions(cascade,v,vt, edges,mentions):
    flag = 0
    #
    for u,t in cascade :
        if t<=vt and v in edges[u]:
        # if t<=vt and v in edges[u] and u in mentions and v in mentions[u] and mentions[u][v] <= vt:
            flag += 1
    return (flag>0)

def gen_gt_cascade(retweets,edges,mentions):
    spreads = set()
    seeds = set()
    orders = {}
    node_step = {}
    for seed in retweets:
        # cascade=[each]
        cur_cascade = set([(seed,0)])
        ans = []
        x = [a for a in retweets[seed].items()]
        x.append((seed, 0))
        candidates = set(x)
        while len(cur_cascade)>0:
            ans.append(cur_cascade)
            candidates = candidates-cur_cascade
            new_order_cascade = set()
            for each in candidates:
                v,t = each
                if check_conditions(cur_cascade,v,t,edges,mentions):
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

def write_gt(seeds,spreads):
    fo = open("twitter.seed", "w")
    for s in seeds:
        fo.write(s+"\n")
    fo = open("twitter.spread", "w")
    for s in spreads:
        fo.write(s+"\n")


if __name__ == '__main__':
    mentionlist,replylist,retweetlist = read_activity("higgs-activity_time.txt")
    show_stats(mentionlist, "mention")
    show_stats(replylist, "reply")
    show_stats(retweetlist, "retweet")
    edgelist = read_edges("twitter.edgelist")
    show_stats(edgelist,"edges")
    seeds, spreads = gen_gt_cascade(retweetlist,edgelist,mentionlist)
    write_gt(seeds,spreads)
