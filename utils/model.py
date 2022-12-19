import random
import numpy as np
import math





def IC_model(args, g, seeds):
    num_of_simulations = args.repeat
    spreads = {}
    step_spread = np.zeros((num_of_simulations,args.step+1), dtype = int)  # spread in each step and overall spread
    for i in range(num_of_simulations):
        # np.random.seed(i)
        new_active, ans = seeds[:], set(seeds[:])
        step = 0
        while new_active:
            #Getting neighbour nodes of newly activate node
            # Calculating if any nodes of those neighbours can be activated, if yes add them to new_ones.
            new_ones = []
            for u in new_active:
                for v in g[u]:
                    if v not in ans:  # check the inactive neighbors of each in new_active
                        if np.random.uniform(0, 1) <= g[u][v]['weight']:
                            ans.add(v)  # activated successfully
                            new_ones.append(v)
            #Checking which ones in new_ones are not in our Ans...only adding them to our Ans so that no duplicate in Ans.
            new_active = new_ones
            # print("new active: ", len(new_active))

            if step<args.step:
                step_spread[i, step] = len(new_active)
            step += 1

        step_spread[i, args.step] = len(ans)-len(seeds)
        for each in ans-set(seeds):
            spreads[each] = 1 if each not in spreads else spreads[each] + 1

    # print("spreads: ",sum(spreads.values()),"real spreads:",np.sum(step_spread[:,args.step]))
    # print("spread in each step (the last is the overall spread): ", )
    eachspread = step_spread[:, args.step]
    value,_ = np.histogram(eachspread, density=True)
    # for i in range(len(value)):
    #     print("({},{})".format(i+1, value[i]/sum(value)))
    sspread = np.mean(step_spread, dtype=float, axis=0)
    step_spread = [(i + 1, sspread[i]) for i in range(len(sspread) - 1)]
    all_spread = sspread[-1]
    active_prob = [(each, spreads[each] / args.repeat) for each in spreads]
    return all_spread, step_spread, active_prob

def ICM_model(args, g, seeds,ddl=15):
    num_of_simulations = args.repeat
    spreads = {}
    step_spread = np.zeros((num_of_simulations,args.step+1), dtype = int)  # spread in each step and overall spread
    for i in range(num_of_simulations):
        # np.random.seed(i)
        new_active, ans = seeds[:], set(seeds[:])
        step = 0
        to_meet_pair = set()
        while step<ddl:
            #Getting neighbour nodes of newly activate node
            # Calculating if any nodes of those neighbours can be activated, if yes add them to new_ones.
            new_ones = []
            tmp = set()
            for u,v in to_meet_pair:
                if np.random.uniform(0, 1) <= g[u][v]['meet']:
                    tmp.add((u, v))  # meet successfully
                    if np.random.uniform(0, 1) <= g[u][v]['weight']:
                        ans.add(v)  # activated successfully
                        new_ones.append(v)
            to_meet_pair -= tmp
            for u in new_active:
                for v in g[u]:
                    if v not in ans:  # check the inactive neighbors of each in new_active
                        if np.random.uniform(0, 1) > g[u][v]['meet']:
                            to_meet_pair.add((u,v)) # keep trying to meet
                        elif np.random.uniform(0, 1) <= g[u][v]['weight']:
                            ans.add(v)  # activated successfully
                            new_ones.append(v)
            #Checking which ones in new_ones are not in our Ans...only adding them to our Ans so that no duplicate in Ans.
            new_active = new_ones
            # print("new active: ", len(new_active))

            if step < args.step:
                step_spread[i, step] = len(new_active)
            step += 1

        step_spread[i, args.step] = len(ans) - len(seeds)
        for each in ans-set(seeds):
            spreads[each] = 1 if each not in spreads else spreads[each] + 1

    # print("spreads: ",sum(spreads.values()),"real spreads:",np.sum(step_spread[:,args.step]))
    eachspread = step_spread[:, args.step]
    value, _ = np.histogram(eachspread, density=True)
    # for i in range(len(value)):
    #     print("({},{})".format(i + 1, value[i]/sum(value)))
    sspread = np.mean(step_spread, dtype=float, axis=0)
    step_spread = [(i + 1, sspread[i]) for i in range(len(sspread) - 1)]
    all_spread = sspread[-1]
    active_prob = [(each, spreads[each] / args.repeat) for each in spreads]
    return all_spread, step_spread, active_prob


def ICN_model(args, g, seeds, q=0.9):
    num_of_simulations = args.repeat
    pos_spreads = {}
    step_pos_spread = np.zeros((num_of_simulations,args.step+1), dtype = int)  # spread in each step and overall spread
    for i in range(num_of_simulations):
        # np.random.seed(i)
        startseed = seeds[:]
        random.shuffle(startseed)
        new_active = {each:1 if np.random.uniform(0, 1) <= q else 0 for each in startseed}  # the role of active nodes: 1 for pos; 0 for neg
        ans = new_active.copy()
        step = 0
        seed_spread=sum(new_active.values())
        while new_active:
            #Getting neighbour nodes of newly activate node
            # Calculating if any nodes of those neighbours can be activated, if yes add them to new_ones.
            new_ones = {}
            for u in new_active:
                for v in g[u]:
                    if v not in ans:  # check the inactive neighbors of each in new_active
                        if np.random.uniform(0, 1) <= g[u][v]['weight']:
                            states = 1 if new_active[u]==1 and np.random.uniform(0, 1) <= q else 0 # new active state transition
                            ans[v] = states
                            new_ones[v] = states
            #Checking which ones in new_ones are not in our Ans...only adding them to our Ans so that no duplicate in Ans.
            new_active = new_ones
            # print("new active: ", len(new_active))

            if step < args.step:
                step_pos_spread[i, step] = sum(new_active.values())
            step += 1

        step_pos_spread[i, args.step] = sum(ans.values()) - seed_spread
        for each in ans.keys()-set(seeds):
            if ans[each]>0:
                pos_spreads[each] = 1 if each not in pos_spreads else pos_spreads[each] + 1

    eachspread = step_pos_spread[:, args.step]
    value, _ = np.histogram(eachspread, density=True)
    # for i in range(len(value)):
    #     print("({},{})".format(i + 1, value[i]/sum(value)))
    sspread = np.mean(step_pos_spread, dtype=float, axis=0)
    step_spread = [(i + 1, sspread[i]) for i in range(len(sspread) - 1)]
    all_spread = sspread[-1]
    active_prob = [(each, pos_spreads[each] / args.repeat) for each in pos_spreads]
    return all_spread, step_spread, active_prob


def ICR_model(args, g, seeds, beta=0.9, gamma=0.6):
    num_of_simulations = args.repeat
    expose_spreads = {}
    step_expose_spread = np.zeros((num_of_simulations,args.step+1), dtype = int)  # spread in each step and overall spread
    for i in range(num_of_simulations):
        # np.random.seed(i)
        startseed = seeds[:]
        random.shuffle(startseed)
        new_active, ans = startseed[:], set(startseed[:])
        new_inviter, expose_ans = startseed[:], set(startseed[:])
        step = 0
        seed_exposer=expose_ans.copy()
        while new_inviter:
            #Getting neighbour nodes of newly activate node
            # Calculating if any nodes of those neighbours can be activated, if yes add them to new_ones.
            new_inviter_ones = []
            new_expose_cnt = 0
            for u in new_inviter:
                for v in g[u]:
                    if v not in ans:  # check the inactive neighbors of each in new_active
                        if np.random.uniform(0, 1) <= g[u][v]['weight']:
                            ans.add(v)
                            if np.random.uniform(0, 1) <= beta:
                                expose_ans.add(v)
                                new_expose_cnt+=1
                                if np.random.uniform(0, 1) <= gamma:
                                    new_inviter_ones.append(v)
            #Checking which ones in new_ones are not in our Ans...only adding them to our Ans so that no duplicate in Ans.
            new_inviter = new_inviter_ones
            # print("new active: ", len(new_active))

            if step < args.step:
                step_expose_spread[i, step] = new_expose_cnt
            step += 1

        step_expose_spread[i, args.step] = len(expose_ans) - len(seed_exposer)
        for each in expose_ans-seed_exposer:
            expose_spreads[each] = 1 if each not in expose_spreads else expose_spreads[each] + 1

    # print("spreads: ",sum(spreads.values()),"real spreads:",np.sum(step_spread[:,args.step]))
    # print("spread in each step (the last is the overall spread): ", np.mean(step_expose_spread, dtype=float, axis=0))
    # res = ["{},{:.4f}".format(each, expose_spreads[each] / args.repeat) for each in expose_spreads]
    eachspread = step_expose_spread[:, args.step]
    value, _ = np.histogram(eachspread, density=True)
    # for i in range(len(value)):
    #     print("({},{})".format(i + 1, value[i]/sum(value)))
    sspread = np.mean(step_expose_spread, dtype=float, axis=0)
    step_spread = [(i + 1, sspread[i]) for i in range(len(sspread) - 1)]
    all_spread = sspread[-1]
    active_prob = [(each, expose_spreads[each] / args.repeat) for each in expose_spreads]
    return all_spread, step_spread, active_prob


def LT_model(args, g, seeds):
    num_of_simulations = args.repeat
    spreads = {}
    step_spread = np.zeros((num_of_simulations,args.step+1), dtype = int)  # spread in each step and overall spread
    for i in range(num_of_simulations):
        new_active, ans = seeds[:], set(seeds[:])
        # np.random.seed(i)
        threshold = {each: np.random.uniform(0,1) for each in g.nodes()} # before the simulation, assign a random threshold to each node
        accumulate = {each: 0 for each in g.nodes()} # the probability summation of activated in-neighbors of each node
        step = 0
        while new_active:
            new_ones = []
            for u in new_active:
                for v in g[u]:
                    if v not in ans: # check the inactive neighbors of each in new_active
                        accumulate[v] += g[u][v]['weight'] # update the summation of probability
                        if accumulate[v] >= threshold[v]:
                            ans.add(v) # activated successfully
                            new_ones.append(v)

            new_active = new_ones
            # print("new active: ", len(new_active))

            if step < args.step:
                step_spread[i, step] = len(new_active)
            step += 1

        step_spread[i, args.step] = len(ans) - len(seeds)
        for each in ans-set(seeds):
            spreads[each] = 1 if each not in spreads else spreads[each] + 1

    eachspread = step_spread[:, args.step]
    value, _ = np.histogram(eachspread, density=True)
    # for i in range(len(value)):
    #     print("({},{})".format(i + 1, value[i]/sum(value)))
    sspread = np.mean(step_spread, dtype=float, axis=0)
    step_spread = [(i + 1, sspread[i]) for i in range(len(sspread) - 1)]
    all_spread = sspread[-1]
    active_prob = [(each, spreads[each] / args.repeat) for each in spreads]
    return all_spread, step_spread, active_prob


def LTC_model(args, g, seeds,lambda_a = 0.3, lambda_b = 0.1, mu_a = 0.001, mu_b = 0.001):
    num_of_simulations = args.repeat
    adopt_spreads = {}
    step_adopt_spread = np.zeros((num_of_simulations,args.step+1), dtype = int)  # spread in each step and overall spread
    for i in range(num_of_simulations):
        # np.random.seed(i)
        threshold = {each: np.random.uniform(0,1) for each in g.nodes()} # before the simulation, assign a random threshold/transition state to each node
        labda = {each: np.random.beta(lambda_a,lambda_b) for each in g.nodes()}
        mu = {each: np.random.beta(mu_a,mu_b) for each in g.nodes()}
        accumulate = {each: 0 for each in g.nodes()} # the probability summation of activated in-neighbors of each node
        # np.random.seed(i)
        new_active = {each: 1 if np.random.uniform(0, 1) <= 0.33 else 0 for each in seeds[:]}  # the role of active nodes: 1 for adoption; 0 for others
        ans = new_active.copy()
        step = 0
        seed_spread = sum(new_active.values())
        while new_active:
            new_ones = {}
            for u in new_active:
                for v in g[u]:
                    if v not in ans: # check the inactive neighbors of each in new_active
                        accumulate[v] += g[u][v]['weight'] # update the summation of probability
                        if accumulate[v] >= threshold[v]:
                            states = 1 if np.random.uniform(0,1) <= labda[v] else 0  # new active state transition
                            ans[v] = states
                            new_ones[v] = states

            new_active = new_ones
            # print("new active: ", len(new_active))

            if step < args.step:
                step_adopt_spread[i, step] = sum(new_active.values())
            step += 1

        step_adopt_spread[i, args.step] = sum(ans.values()) - seed_spread
        for each in ans.keys()-set(seeds):
            if ans[each] > 0:
                adopt_spreads[each] = 1 if each not in adopt_spreads else adopt_spreads[each] + 1

    eachspread = step_adopt_spread[:, args.step]
    value, _ = np.histogram(eachspread, density=True)
    # for i in range(len(value)):
    #     print("({},{})".format(i + 1, value[i]/sum(value)))
    sspread = np.mean(step_adopt_spread, dtype=float, axis=0)
    step_spread = [(i + 1, sspread[i]) for i in range(len(sspread) - 1)]
    all_spread = sspread[-1]
    active_prob = [(each, adopt_spreads[each] / args.repeat) for each in adopt_spreads]
    return all_spread, step_spread, active_prob


def FTM_model(args, g, seeds):
    num_of_simulations = args.repeat
    spreads = {}
    step_spread = np.zeros((num_of_simulations,args.step+1), dtype = int)  # spread in each step and overall spread
    for i in range(num_of_simulations):
        new_active, ans = seeds[:], set(seeds[:])
        # np.random.seed(i)
        threshold = {each: np.random.uniform(0,1) for each in g.nodes()} # before the simulation, assign a random threshold to each node
        accumulate = {each: 0 for each in g.nodes()} # the probability summation of activated in-neighbors of each node
        step = 0
        while new_active:
            new_ones = []
            for u in new_active:
                for v in g[u]:
                    if v not in ans: # check the inactive neighbors of each in new_active
                        accumulate[v] += g[u][v]['weight'] # update the summation of probability
                        expkernel = math.exp(accumulate[v])/(1+math.exp(accumulate[v]))
                        if expkernel >= threshold[v]:
                            ans.add(v) # activated successfully
                            new_ones.append(v)

            new_active = new_ones
            # print("new active: ", len(new_active))

            if step < args.step:
                step_spread[i, step] = len(new_active)
            step += 1

        step_spread[i, args.step] = len(ans) - len(seeds)
        for each in ans-set(seeds):
            spreads[each] = 1 if each not in spreads else spreads[each] + 1

    # print("spreads: ",sum(spreads.values()),"real spreads:",np.sum(step_spread[:,args.step]))
    # print("spread in each step (the last is the overall spread): ", np.mean(step_spread, dtype=float, axis=0))
    # res = ["{},{:.4f}".format(each, spreads[each] / args.repeat) for each in spreads]
    eachspread = step_spread[:, args.step]
    value, _ = np.histogram(eachspread, density=True)
    # for i in range(len(value)):
    #     print("({},{})".format(i + 1, value[i]/sum(value)))
    sspread = np.mean(step_spread, dtype=float, axis=0)
    step_spread = [(i + 1, sspread[i]) for i in range(len(sspread) - 1)]
    all_spread = sspread[-1]
    active_prob = [(each, spreads[each] / args.repeat) for each in spreads]
    return all_spread, step_spread, active_prob


