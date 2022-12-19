import numpy as np

from utils.util import *
from utils.model import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="ic",
        type=str
    )
    parser.add_argument(
        "--dataset",
        default="digg",
        type=str
    )
    parser.add_argument(
        "--output",
        default=1,
        type=int
    )
    parser.add_argument(
        "--repeat",
        default=100,
        type=int
    )
    parser.add_argument(
        "--step",
        default=10,
        type=int
    )
    parser.add_argument(
        "--beta",
        default=0.9,
        type=float
    )
    parser.add_argument(
        "--gamma",
        default=0.6,
        type=float
    )
    args, _ = parser.parse_known_args()

    g = load_graph("datasets/{}.edgelist".format(args.dataset))
    seeds = load_seeds("datasets/{}.seed".format(args.dataset))
    true_active = load_seeds("datasets/{}.spread".format(args.dataset))
    print(args)

    all_spread, step_spread, active_prob = None,None,None
    if args.model == 'ic': # input data  1/d
        all_spread, step_spread, active_prob = IC_model(args, g, seeds)
    elif args.model == 'icm': # input data p=1/d, default param m(u,v)=5/(5+d_out(u)), ddl=15
        all_spread, step_spread, active_prob = ICM_model(args, g, seeds)
    elif args.model == 'icn': # input data p=1/d, default param q=0.9
        all_spread, step_spread, active_prob = ICN_model(args, g, seeds)
    elif args.model == 'icr': # input data p=1/d, default param beta=0.9, gamma=0.6
        all_spread, step_spread, active_prob = ICR_model(args, g, seeds,beta=args.beta,gamma=args.gamma)
    elif args.model == 'lt': # input data p=1/d
        all_spread, step_spread, active_prob = LT_model(args, g, seeds)
    elif args.model == 'ftm': # input data p=1/d
        all_spread, step_spread, active_prob = FTM_model(args, g, seeds)
    elif args.model == 'ltc': # input data p=1/d, default param lambda_a = 0.3 lambda_b = 0.1 mu_a = 0.001 mu_b = 0.001
        all_spread, step_spread, active_prob = LTC_model(args, g, seeds)
    else:
        exit("Unsupported model")
    print("--------------------Results--------------------------")
    print("overall spreads:")
    print(all_spread)
    print("step spreads:")
    print(step_spread)
    print("activation probability of each user:")
    print(active_prob)



