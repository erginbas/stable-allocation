# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import uuid
from multiprocessing import Pool
import itertools
import logging
import argparse

from data_utils import get_dataset, get_capacity
from algorithms import Algorithms

""" show plots """
show_plots = False

""" number of monte carlo simulations """
num_sims = 20

"""verbose"""
verbose = True

"""save location"""
save_path = 'results'


def run_exp(input):

    exp_id, dataset, structure, is_dynamic = input[0]
    iteration = input[1]

    data_arg = {}
    if dataset == "synthetic":
        T = 400
        if is_dynamic:
            data_arg["N"] = 350
            data_arg["M"] = 50
        else:
            data_arg["N"] = 250
            data_arg["M"] = 200
        data_arg["rank"] = 10
    elif dataset == "restaurant":
        T = 400
    elif dataset == "yelp":
        T = 600
    elif dataset == "movie":
        T = 400
    else:
        raise NotImplementedError

    R_true, rank, V_true = get_dataset(dataset, **data_arg)

    np.random.seed(np.random.randint(1, 1000) + iteration)

    N, M = R_true.shape

    # noise sub-gaussianity parameter
    eta = 0.2

    p_activity = 0.2
    cap_to_dem_ratio = 0.6
    C, D = get_capacity(N, M, T, is_dynamic, cap_to_dem_ratio=cap_to_dem_ratio, p_activity=p_activity)

    # save the simulation configutarion
    exp_params = {"N": N, "M": M, "T": T, "dataset": dataset, "rank": rank, "structure": structure,
                  "C_max": int(np.max(C)), "p_activity": p_activity, "dynamic": int(is_dynamic)}

    exp_save_path = f"{save_path}/{exp_id}/{iteration}"
    Path(exp_save_path).mkdir(exist_ok=True, parents=True)
    with open(f"{exp_save_path}/params.json", "w") as outfile:
        json.dump(exp_params, outfile)

    logging.basicConfig(filename=f"{exp_save_path}/info.log",
                        filemode='a',
                        format='[%(asctime)s] %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    # print summary of parameters
    logging.info(f"T = {T}")
    logging.info(f"N = {N},  M = {M}")
    logging.info(f"total C = {np.sum(C[0])}")
    logging.info(f"total D = {np.sum(D[0])}")

    """
    LR-ILAP: Allocations and pricing with low-rank collaborative filtering (our proposed algorithm)
    CX-ILAP: Allocations and pricing with known contextual information (our proposed algorithm)
    LR-RWE: Low-rank collaborative filtering without exploration (choosing the best allocation w.r.t. LS estimate)
    CX-RWE: Contextual collaborative filtering without exploration (choosing the best allocation w.r.t. LS estimate)
    LR-IR: Interactive low-rank collaborative filtering (does not consider the capacity constraints, 
    but obtains zero reward when an item cannot be allocated due to exceeding the capacity)
    CX-IR: Interactive contextual collaborative filtering (does not consider the capacity constraints, 
    but obtains zero reward when an item cannot be allocated due to exceeding the capacity)
    CUCB: Allocations and pricing without using any structural information
    """

    alg_helper = Algorithms(R_true, V_true, rank, eta, is_dynamic, C, D, T, exp_save_path)

    alg_list = [f"{structure}-ILAP", "CUCB", f"{structure}-RWE", f"{structure}-IR"]
    regrets_list = []

    for alg in alg_list:
        regrets_list.append(alg_helper.solve_algo(alg))

    ''' Plots '''
    if show_plots:
        for i, alg in enumerate(alg_list):
            plt.plot(np.arange(T), regrets_list[i][:, 0], label=alg)
        plt.ylabel('Regret')
        plt.xlabel('Iteration')
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('params', metavar='N', type=str, nargs='+')

    args = parser.parse_args()

    """choose data set (synthetic, movie, restaurant)"""
    dataset = args.params[0]
    """set LR (low-rank) or CX (contextual)"""
    structure = args.params[1]
    """set if dynamic/static"""
    if args.params[2] == "dynamic":
        is_dynamic = True
    elif args.params[2] == "static":
        is_dynamic = False
    else:
        raise ValueError

    # set a unique ID for the experiment
    if args.params[3]:
        exp_id = args.params[3]
    else:
        exp_id = uuid.uuid4().hex[:6]

    print(f"ID: {exp_id}")
    params = (exp_id, dataset, structure, is_dynamic)

    query_inputs = list(zip(itertools.repeat(params), range(num_sims)))

    with Pool(processes=10) as pool:
        results = list(tqdm(pool.imap(run_exp, query_inputs), total=num_sims))
