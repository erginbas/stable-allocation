import numpy as np
import scipy.optimize as opt


def conf_projection_u(U_hat, V, Theta_ls, counts, beta):
    if len(U_hat.shape) == 1:
        U_hat = U_hat[np.newaxis, :]
        Theta_ls = Theta_ls[np.newaxis, :]
        counts = counts[np.newaxis, :]

    N, _ = U_hat.shape
    M, _ = V.shape

    W_list = counts
    G_list = np.einsum('rj,ir,rk->ijk', V, W_list, V, optimize="greedy")
    G_inv = np.linalg.inv(G_list)
    S_list, Q_list = np.linalg.eig(G_list)
    Q_list = np.swapaxes(Q_list, 1, 2)
    Q_U_hat = np.einsum('ijk,ik->ij', np.array(Q_list), U_hat, optimize="greedy")
    W_Theta = W_list * Theta_ls
    b = np.einsum('ikm,rm,ir->ik', G_inv, V, W_Theta, optimize="greedy")
    gamma = beta + np.sum(Theta_ls * counts * (b @ V.T - Theta_ls))
    z = np.einsum('ijk,ik->ij', Q_list, b, optimize="greedy")

    def func(lmb):
        return np.sum(S_list * (Q_U_hat - z)**2 / (lmb * S_list + 1)**2) - gamma

    lmb_star = opt_func(func, gamma)

    Q_U_star = (lmb_star * S_list * z + Q_U_hat)/(lmb_star * S_list + 1)
    U_star = np.einsum('ikj,ik->ij', Q_list, Q_U_star, optimize="greedy")

    return U_star


def conf_projection_u_inf(U_hat, V, Theta_ls, counts, rho):
    if len(U_hat.shape) == 1:
        U_hat = U_hat[np.newaxis, :]
        Theta_ls = Theta_ls[np.newaxis, :]
        counts = counts[np.newaxis, :]

    N, _ = U_hat.shape
    M, _ = V.shape

    W_list = counts
    G_list = np.einsum('rj,ir,rk->ijk', V, W_list, V, optimize="greedy")
    G_inv = np.linalg.inv(G_list)
    S_list, Q_list = np.linalg.eig(G_list)
    Q_list = np.swapaxes(Q_list, 1, 2)
    Q_U_hat = np.einsum('ijk,ik->ij', np.array(Q_list), U_hat, optimize="greedy")
    W_Theta = W_list * Theta_ls
    b = np.einsum('ikm,rm,ir->ik', G_inv, V, W_Theta, optimize="greedy")
    gamma = rho + np.sum(Theta_ls * counts * (b @ V.T - Theta_ls), axis=1)
    z = np.einsum('ijk,ik->ij', Q_list, b, optimize="greedy")

    lmb_star = np.zeros(N)

    for i in range(N):
        def func(lmb):
            return np.sum(S_list[i] * (Q_U_hat[i] - z[i])**2 / (lmb * S_list[i] + 1)**2) - gamma[i]

        lmb_star[i] = opt_func(func, gamma[i])

    Q_U_star = ((S_list.T * z.T * lmb_star + Q_U_hat.T)/(S_list.T * lmb_star + 1)).T
    U_star = np.einsum('ikj,ik->ij', Q_list, Q_U_star, optimize="greedy")

    return U_star

def opt_func(func, gamma):
    if gamma < 0:
        lmb_star = 0
    elif func(0) < 0:
        lmb_star = 0
    else:
        lmb_min = 0
        lmb_max = 1
        while True:
            if func(lmb_max) < 0:
                break
            lmb_min = lmb_max
            lmb_max = lmb_max * 2

        lmb_star = opt.bisect(func, lmb_min, lmb_max)
    return lmb_star



def conf_projection_v(V_hat, U, Theta_ls, counts, beta):
    return conf_projection_u(V_hat, U, Theta_ls.T, counts.T, beta)


if __name__ == '__main__':
    N = 50
    M = 10
    R = 5

    U_hat = np.random.rand(N, R)
    V = np.random.rand(M, R)
    Theta_ls = np.random.rand(N, M)
    counts = np.ones((N, M))

    beta_list = np.arange(5, 50)

    for beta in beta_list:
        U_star = conf_projection_u(U_hat, V, Theta_ls, counts, beta)
        print(f"beta = {beta}")
        if U_star is None:
            print("not feasible")
        else:
            UV_star = U_star @ V.T
            print(np.sum((U_star - U_hat) ** 2))
            print(np.sum(counts * (UV_star - Theta_ls) ** 2))
        print()
