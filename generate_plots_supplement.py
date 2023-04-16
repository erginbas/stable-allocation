import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from matplotlib import rc

plt.style.use(['seaborn-deep', 'paper.mplstyle'])
matplotlib.rcParams.update({"axes.grid": False})

# # matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'text.usetex': True,
#     'pgf.rcfonts': True,
# })

rc('font', **{'family': 'serif', 'serif': ['Nimbus Roman No9 L']})

all_exp_id = [["f96dd7", "13395313"], ["13496587", "13496586"], ["13439788", "13418043"], ["13491005", "13439790"]]

names = ["ILAP (Proposed Algorithm)", "RWE", "IR", "CUCB"]
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors_to_use = color_cycle[:len(names)]

for pairs_i, exp_id_pairs in enumerate(all_exp_id):

    cols_exp_id = [exp_id_pairs[0], exp_id_pairs[0], exp_id_pairs[1], exp_id_pairs[1]]
    cols_accept_reject = [False, True, False, True]
    num_of_runs = 8

    fig, ax = plt.subplots(2, 4, figsize=(6.9, 2.7))
    fig.tight_layout(w_pad=0.2, h_pad=0.4, rect=(0.01, 0.01, 0.998, 0.86))
    fig.subplots_adjust(wspace=0.4, hspace=0.3)


    legendLines = []

    for exp_i, (experiment_id, accept_reject) in enumerate(zip(cols_exp_id, cols_accept_reject)):
        with open(f"final_results/{experiment_id}/{0}/params.json", 'r') as f:
            data = json.load(f)

        T = data["T"]
        structure = data["structure"]
        is_dynamic = data["dynamic"]

        if data["dataset"] == "synthetic":
            dataset = "Synthetic dataset"
            if is_dynamic:
                N = 350
                M = 50
                R = 10
            else:
                N = 250
                M = 200
                R = 20
        elif data["dataset"] == "movie":
            dataset = "MovieLens dataset"
            N, M = 650, 450
            R = 50
        if data["dataset"] == "yelp":
            dataset = "Yelp dataset"
            N, M = 700, 300
            R = 60

        methods = [f"{structure}-ILAP", f"{structure}-RWE", f"{structure}-IR", "CUCB"]
        num_of_methods = len(methods)

        regrets = np.zeros((num_of_methods, num_of_runs, T, 4))
        opt_rewards = np.zeros((num_of_runs, T))

        for i in range(num_of_runs):
            for m, method in enumerate(methods):
                regrets[m, i] = np.load(f"final_results/{experiment_id}/{i}/{method}_regrets.npy")
            opt_rewards[i] = np.load(f"final_results/{experiment_id}/{i}/opt_rewards.npy")

        mean_opt_rewards = np.mean(opt_rewards, axis=0)
        mean_regrets = np.mean(regrets, axis=1)
        std_regrets = np.std(regrets, axis=1)
        mean_rewards = mean_opt_rewards[np.newaxis, :, np.newaxis] - mean_regrets

        if accept_reject:
            chosen_plots = [2, 3]
        else:
            chosen_plots = [0, 1]

        for j in range(num_of_methods):
            line, = ax[0, exp_i].plot(np.linspace(1, T + 1, T), mean_rewards[j, :, chosen_plots[0]],
                              label=names[j], color=colors_to_use[j], zorder=10*(num_of_methods - j))
            legendLines.append(line)
            ax[0, exp_i].fill_between(np.linspace(1, T + 1, T),
                                      mean_rewards[j, :, chosen_plots[0]] - std_regrets[j, :, chosen_plots[0]],
                               mean_rewards[j, :, chosen_plots[0]] + std_regrets[j, :, chosen_plots[0]],
                                      color=colors_to_use[j], alpha=0.2, zorder=10*(num_of_methods - j))

        line, = ax[0, exp_i].plot(np.linspace(1, T + 1, T), mean_opt_rewards, color="black", linestyle="dashed")
        legendLines.append(line)

        ax[0, exp_i].set(xlabel='Iteration', ylabel='Social Welfare Reward')
        ax[0, exp_i].grid()
        ax[0, exp_i].set_ylim(bottom=1e-4)
        ax[0, exp_i].set_xlim(left=1e-4)
        title = f'{"contextual" if structure == "CX" else "low-rank"}\n' \
                f'{"with" if accept_reject else "without"} rejections'
        ax[0, exp_i].set_title(title, x=0.5, y=1.0, pad=4)
        ax[0, exp_i].set_xticks(np.arange(0, T + 1, 100))

        for j in range(num_of_methods):
            ax[1, exp_i].plot(np.linspace(1, T + 1, T), mean_regrets[j, :, chosen_plots[1]],
                              label=names[j], color=colors_to_use[j], zorder=10*(num_of_methods - j))
            ax[1, exp_i].fill_between(np.linspace(1, T + 1, T),
                                      mean_regrets[j, :, chosen_plots[1]] - std_regrets[j, :, chosen_plots[1]],
                               mean_regrets[j, :, chosen_plots[1]] + std_regrets[j, :, chosen_plots[1]],
                                      color=colors_to_use[j], alpha=0.2, zorder=10*(num_of_methods - j))

        ax[1, exp_i].set(xlabel='Iteration', ylabel='Instability')
        ax[1, exp_i].grid()
        ax[1, exp_i].set_ylim(bottom=1e-4)
        ax[1, exp_i].set_xlim(left=1e-4)
        ax[1, exp_i].set_xticks(np.arange(0, T + 1, 100))

    leg = fig.legend(legendLines[:len(names) + 1], names + ["Optimum Social Welfare"], loc='upper center',
                     frameon=True, ncol=len(names) + 1, columnspacing=1, prop=dict(size=7))
    leg_lines = leg.get_lines()
    for line in leg_lines:
        plt.setp(line, linewidth=2)

    plt.savefig(f"plots/pdfs/exp_supp_{pairs_i}.pdf")
    plt.savefig(f"plots/exp_supp_{pairs_i}.jpeg")
