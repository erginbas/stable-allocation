import numpy as np
import matplotlib.pyplot as plt


def get_synthetic_dataset(**kwargs):
    N = kwargs["N"]
    M = kwargs["M"]

    # rank of R
    rank = kwargs["rank"]

    # generate low-rank R
    U = np.random.randn(N, rank - 1)
    U = (U.T / np.linalg.norm(U, axis=1)).T

    V = np.random.randn(M, rank - 1)
    V = (V.T / np.linalg.norm(V, axis=1)).T

    R_true = (U @ V.T + np.ones((N, M)))/2

    u, s, vh = np.linalg.svd(R_true)
    V_true = vh[:rank, :].T @ np.diag(np.sqrt(s[:rank]))

    if kwargs.get("plot_data", False):
        plt.hist(R_true.flatten(), bins=np.int32(np.sqrt(N * M)))
        _, s, _ = np.linalg.svd(R_true)
        plt.yscale("log")
        plt.plot(s)
        plt.show()

    return R_true, rank, V_true


def get_restaurant_dataset(**kwargs):
    R_true = np.load("data/rating_restaurant_completed.npy")
    # normalize maximum value of R
    R_true = R_true / 6
    R_true = R_true + 0.03 * np.random.normal(size=R_true.shape)

    # rank of R
    rank = 10

    N, M = R_true.shape

    u, s, vh = np.linalg.svd(R_true)
    V_true = vh[:rank, :].T @ np.diag(np.sqrt(s[:rank]))

    if kwargs.get("plot_data", False):
        plt.hist(R_true.flatten(), bins=np.int32(np.sqrt(N * M)))
        _, s, _ = np.linalg.svd(R_true)
        plt.yscale("log")
        plt.plot(s)
        plt.show()

    return R_true, rank, V_true


def get_movie_dataset(**kwargs):
    R_true = np.load("data/rating_ml_100k_completed.npy")

    R_true = R_true[:300, :200]

    # normalize maximum value of R
    R_true = R_true / np.max(R_true)

    R_true = R_true + 0.03 * np.random.normal(size=R_true.shape)

    # rank of R
    rank = 20

    N, M = R_true.shape

    u, s, vh = np.linalg.svd(R_true)
    V_true = vh[:rank, :].T @ np.diag(np.sqrt(s[:rank]))

    if kwargs.get("plot_data", False):
        plt.hist(R_true.flatten(), bins=np.int32(np.sqrt(N * M)))
        _, s, _ = np.linalg.svd(R_true)
        plt.yscale("log")
        plt.plot(s)
        plt.show()

    return R_true, rank, V_true

def get_yelp_dataset(**kwargs):
    R_true = np.load("data/rating_yelp_completed.npy")

    # normalize maximum value of R
    R_true = R_true / np.max(R_true)

    R_true = R_true[:500, :100]

    # rank of R
    rank = 20

    N, M = R_true.shape

    u, s, vh = np.linalg.svd(R_true)
    V_true = vh[:rank, :].T @ np.diag(np.sqrt(s[:rank]))

    if kwargs.get("plot_data", False):
        plt.hist(R_true.flatten(), bins=np.int32(np.sqrt(N * M)))
        _, s, _ = np.linalg.svd(R_true)
        plt.yscale("log")
        plt.plot(s)
        plt.show()

    return R_true, rank, V_true


def get_dataset(dataset=None, **kwargs):
    if dataset == "synthetic":
        return get_synthetic_dataset(**kwargs)
    if dataset == "restaurant":
        return get_restaurant_dataset(**kwargs)
    if dataset == "yelp":
        return get_yelp_dataset(**kwargs)
    if dataset == "movie":
        return get_movie_dataset(**kwargs)


def get_capacity(N, M, T, is_dynamic, cap_to_dem_ratio=1, p_activity=0.2):
    if is_dynamic:
        D = np.random.choice(2, size=(T, N), p=[1 - p_activity, p_activity])
        # set capacities randomly (changing with time)
        C_max = int(np.ceil(cap_to_dem_ratio * np.sum(D[0]) / M))
        C = np.zeros((T, M), dtype=np.int64)
        C[0] = np.random.choice(C_max, size=(M)) + 1
        for t in range(1, T):
            C[t] = np.clip(C[t - 1] + np.random.choice(3, size=(M), p=[0.05, 0.9, 0.05]) - 1, 0, 2 * C_max)
        print(np.sum(C, axis=1))
    else:
        # set demands to 1 for all users
        D = np.ones((T, N), dtype=np.int64)
        # set capacities randomly (fixed in time)
        C_max = int(np.ceil(cap_to_dem_ratio * np.sum(D[0]) / M))
        C = np.zeros((T, M), dtype=np.int64)
        C[0] = np.random.choice(C_max, size=(M)) + 1
        for t in range(1, T):
            C[t] = C[t - 1]
    return C, D