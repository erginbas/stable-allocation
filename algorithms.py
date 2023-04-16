import numpy as np
from PyomoSolver import PyomoSolver
import time
import logging
from opt_utils import conf_projection_u, conf_projection_v, conf_projection_u_inf

class Algorithms:
    def __init__(self, R_true, V_true, rank, eta, is_dynamic, C, D, T, exp_save_path, verbose = False):
        self.R_true = R_true
        self.V_true = V_true
        self.rank = rank
        self.N, self.M = R_true.shape
        self.eta = eta
        self.T = T
        self.is_dynamic = is_dynamic
        self.C = C
        self.D = D
        self.exp_save_path = exp_save_path

        logging.basicConfig(filename=f"{exp_save_path}/info.log",
                            filemode='a',
                            format='[%(asctime)s] %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

        self.initial_mask = None
        self.opt_rewards = None

        # use a MIP solver to calculate optimal allocations efficiently
        self.solver = PyomoSolver(self.N, self.M)
        self.find_optimum()
        self.generate_initial_mask()


    def find_optimum(self):
        # calculate optimal allocations for each time t
        self.x_star = np.zeros((self.T, self.N, self.M))
        self.opt_rewards = np.zeros(self.T)

        # solve for optimum allocations
        if self.is_dynamic:
            for t in range(self.T):
                self.x_star[t] = self.solver.solve_system(self.R_true, self.C[t], self.D[t])
                self.opt_rewards[t] = np.sum(self.x_star[t] * self.R_true)
                if t % 10 == 0:
                    logging.info(f'solved x_star at {t}, optimum value = {np.sum(self.x_star[t] * self.R_true)}')
        else:
            for t in range(self.T):
                if t == 0:
                    self.x_star[0] = self.solver.solve_system(self.R_true, self.C[0], self.D[0])
                    self.opt_rewards[t] = np.sum(self.x_star[t] * self.R_true)
                    logging.info(f'solved x_star, optimum value = {np.sum(self.x_star[t] * self.R_true)}')
                else:
                    self.x_star[t] = self.x_star[t - 1]
                    self.opt_rewards[t] = np.sum(self.x_star[t] * self.R_true)

        logging.info(f"Optimum prices = {self.solver.get_prices()}")

        np.save(f"{self.exp_save_path}/opt_rewards", self.opt_rewards)

    def calculate_instability(self, t, X, p):
        best_surplus_list = np.zeros(self.N)
        total_current_surplus = np.sum(X * (self.R_true - p))

        for u in range(self.N):
            demand_u = self.D[t, u]
            if demand_u > 0:
                ordered_items = np.maximum(np.sort(self.R_true[u, :] - p), 0)
                best_surplus_list[u] = np.sum(ordered_items[-demand_u:])

        return np.sum(best_surplus_list) - total_current_surplus

    def calculate_regret(self, t, X, p_walrasian, nu):
        sw_regret = self.opt_rewards[t] - np.sum(X * self.R_true)
        instability = self.calculate_instability(t, X, p_walrasian)

        p_shifted = p_walrasian - nu
        accepted_offers = X * (self.R_true > p_shifted)
        sw_regret_ar = self.opt_rewards[t] - np.sum(accepted_offers * self.R_true)
        instability_ar = self.calculate_instability(t, accepted_offers, p_shifted)

        return [sw_regret, instability, sw_regret_ar, instability_ar]

    def generate_initial_mask(self):
        # generate initial mask
        p = 0.06
        is_mask_feasible = False
        while not is_mask_feasible:
            initial_mask = np.random.binomial(1, p, size=(self.N, self.M))
            p += 0.01
            if np.min(np.sum(initial_mask, axis=0)) > 0 and np.min(np.sum(initial_mask, axis=1)) > 0:
                is_mask_feasible = True
                self.initial_mask = initial_mask

    def solve_without_cap(self, R, t):
        X = np.zeros((self.N, self.M))
        for u in range(self.N):
            demand_u = self.D[t, u]
            if demand_u > 0:
                ordered_items = np.argsort(R[u, :])
                X[u, ordered_items[-demand_u:]] = 1
        return X

    def solve_algo(self, algorithm):
        if algorithm == "CUCB":
            return self.solve_CUCB()
        elif algorithm == "LR-ILAP":
            return self.solve_LR_ILAP()
        elif algorithm == "CX-ILAP":
            return self.solve_CX_ILAP()
        elif algorithm == "LR-RWE":
            return self.solve_LR_RWE()
        elif algorithm == "CX-RWE":
            return self.solve_CX_RWE()
        elif algorithm == "LR-IR":
            return self.solve_LR_IR()
        elif algorithm == "CX-IR":
            return self.solve_CX_IR()
        else:
            raise NotImplementedError

    def solve_CUCB(self):
        """### CUCB (Combinatorial UCB for CMAB without structure)"""

        alpha = 0.3

        # shift parameter for prices (only to be used if users accept/reject)
        nu = 0.6 * ((self.eta ** 2)/(self.N * self.M)) ** (1 / 4)
        logging.info(f"Price shift coeff.= {nu}")

        def get_ucb(observations):
            data_avg, data_count = observations
            std = alpha / np.sqrt(data_count)
            return np.minimum(np.maximum(data_avg + std, 0), 1)

        regrets = np.zeros((self.T, 4))

        # get initial observations
        observations = self.get_observations(self.initial_mask)

        x_prev = None

        logging.info(f'Running CUCB')

        for t in range(self.T):

            R_UCB = get_ucb(observations)

            x_UCB = self.solver.solve_system(R_UCB, self.C[t], self.D[t], x_prev=x_prev)
            x_prev = x_UCB.copy()

            data_avg, counts = observations
            w_t = np.sum(x_UCB/counts)

            # print(np.sum(x_UCB * counts), nu * np.sqrt(w_t))

            p_wal = self.solver.get_prices()
            regret_collection = self.calculate_regret(t, x_UCB, p_wal, nu * np.sqrt(w_t))
            regrets[t] = regret_collection

            if t % 10 == 0:
                logging.info(f'Iter {t}')
                logging.info(f'Social Welfare Regret = {regrets[t, 0]}')
                logging.info(f'Instability = {regrets[t, 1]}')
                logging.info(f'Social Welfare Regret (A/R) = {regrets[t, 2]}')
                logging.info(f'Instability (A/R) = {regrets[t, 3]}')

            observations = self.get_observations(x_UCB, observations)

        self.save_results("CUCB", regrets)
        return regrets

    def solve_LR_ILAP(self):
        """our proposed algorithm, OFU with low-rank collaborative filtering"""

        logging.info(f'Running LR-ILAP')

        # shift parameter for prices (only to be used if users accept/reject)
        nu = 0.6 * (((self.N + self.M) * self.rank * self.eta ** 2) / (self.N ** 2 * self.M ** 2)) ** (1 / 4)
        logging.info(f"Price shift coeff. = {nu}")

        # get initial observations
        observations = self.get_observations(self.initial_mask)

        regrets = np.zeros((self.T, 4))
        x_prev = None

        for t in range(self.T):
            U, V = self.compute_center(observations)
            data_avg, counts = observations

            x_OFU = self.solve_for_ofu_allocation(U, V, counts, t, x_prev=x_prev)
            x_prev = x_OFU.copy()

            w_t = np.sum(x_OFU/counts)

            # print(np.sum(x_OFU * counts), nu * np.sqrt(w_t))

            p_wal = self.solver.get_prices()
            regret_collection = self.calculate_regret(t, x_OFU, p_wal, nu * np.sqrt(w_t))
            regrets[t] = regret_collection

            if t % 10 == 0:
                logging.info(f'Iter {t}')
                logging.info(f'Social Welfare Regret = {regrets[t, 0]}')
                logging.info(f'Instability = {regrets[t, 1]}')
                logging.info(f'Social Welfare Regret (A/R) = {regrets[t, 2]}')
                logging.info(f'Instability (A/R) = {regrets[t, 3]}')

            observations = self.get_observations(x_OFU, observations)

        self.save_results("LR-ILAP", regrets)
        return regrets

    def solve_CX_ILAP(self):
        """our proposed algorithm, OFU with linear contextual bandits"""

        logging.info(f'Running CX-ILAP')

        # shift parameter for prices (only to be used if users accept/reject)
        nu = 0.6 * ((self.rank * self.eta ** 2) / (self.N * self.M ** 2)) ** (1 / 4)
        logging.info(f"Price shift coeff.= {nu}")

        # get initial observations
        observations = self.get_observations(self.initial_mask)

        regrets = np.zeros((self.T, 4))
        x_prev = None

        for t in range(self.T):
            U = self.compute_center_contextual(observations)
            data_avg, counts = observations

            x_OFU = self.solve_for_contextual_ofu_allocation(U, counts, t, x_prev=x_prev)
            x_prev = x_OFU.copy()

            w_t = np.sum(x_OFU/counts)

            p_wal = self.solver.get_prices()
            regret_collection = self.calculate_regret(t, x_OFU, p_wal, nu * np.sqrt(w_t))
            regrets[t] = regret_collection

            if t % 10 == 0:
                logging.info(f'Iter {t}')
                logging.info(f'Social Welfare Regret = {regrets[t, 0]}')
                logging.info(f'Instability = {regrets[t, 1]}')
                logging.info(f'Social Welfare Regret (A/R) = {regrets[t, 2]}')
                logging.info(f'Instability (A/R) = {regrets[t, 3]}')

            observations = self.get_observations(x_OFU, observations)

        self.save_results("CX-ILAP", regrets)
        return regrets

    def solve_LR_RWE(self):
        """ RWE (Low-rank collaborative filtering without exploration, best allocation w.r.t. LS estimate)"""
        # get initial observations
        observations = self.get_observations(self.initial_mask)

        regrets = np.zeros((self.T, 4))

        logging.info(f'Running RWE')

        x_prev = None

        for t in range(self.T):

            U, V = self.compute_center(observations)
            R_est = U @ V.T

            x_t = self.solver.solve_system(R_est, self.C[t], self.D[t], x_prev=x_prev)
            x_prev = x_t.copy()

            p_wal = self.solver.get_prices()
            regret_collection = self.calculate_regret(t, x_t, p_wal, 0)
            regrets[t] = regret_collection

            if t % 10 == 0:
                logging.info(f'Iter {t}')
                logging.info(f'Social Welfare Regret = {regrets[t, 0]}')
                logging.info(f'Instability = {regrets[t, 1]}')
                logging.info(f'Social Welfare Regret (A/R) = {regrets[t, 2]}')
                logging.info(f'Instability (A/R) = {regrets[t, 3]}')

            observations = self.get_observations(x_t, observations)

        self.save_results("LR-RWE", regrets)
        return regrets

    def solve_CX_RWE(self):
        """ RWE (Low-rank collaborative filtering without exploration, best allocation w.r.t. LS estimate)"""
        # get initial observations
        observations = self.get_observations(self.initial_mask)

        regrets = np.zeros((self.T, 4))

        logging.info(f'Running CX-RWE')

        x_prev = None

        for t in range(self.T):

            U = self.compute_center_contextual(observations)
            R_est = U @ self.V_true.T

            x_t = self.solver.solve_system(R_est, self.C[t], self.D[t], x_prev=x_prev)
            x_prev = x_t.copy()

            p_wal = self.solver.get_prices()
            regret_collection = self.calculate_regret(t, x_t, p_wal, 0)
            regrets[t] = regret_collection

            if t % 10 == 0:
                logging.info(f'Iter {t}')
                logging.info(f'Social Welfare Regret = {regrets[t, 0]}')
                logging.info(f'Instability = {regrets[t, 1]}')
                logging.info(f'Social Welfare Regret (A/R) = {regrets[t, 2]}')
                logging.info(f'Instability (A/R) = {regrets[t, 3]}')

            observations = self.get_observations(x_t, observations)

        self.save_results("CX-RWE", regrets)
        return regrets

    def solve_LR_IR(self):

        # get initial observations
        observations = self.get_observations(self.initial_mask)

        logging.info(f'Running LR-IR')

        regrets = np.zeros((self.T, 4))

        x_prev = None

        for t in range(self.T):

            U, V = self.compute_center(observations)
            data_avg, counts = observations

            # obtain solution without capacity constraints
            x_OFU_intended = self.solve_for_ofu_allocation(U, V, counts, t, x_prev=x_prev, solve_with_capacity=False)
            x_OFU = x_OFU_intended.copy()

            # only some of the users obtain the items if there are too many requests for the same item
            for i in range(self.M):
                while np.sum(x_OFU[:, i]) > self.C[t, i]:
                    allocated_users = x_OFU[:, i] > 0.9
                    v = np.random.choice(np.where(allocated_users)[0])
                    x_OFU[v, i] = 0

            x_OFU_uns = x_OFU_intended - x_OFU

            p = np.zeros(self.M)
            regret_collection = self.calculate_regret(t, x_OFU, p, 0)
            regrets[t] = regret_collection

            if t % 10 == 0:
                logging.info(f'Iter {t}')
                logging.info(f'Social Welfare Regret = {regrets[t, 0]}')
                logging.info(f'Instability = {regrets[t, 1]}')
                logging.info(f'Social Welfare Regret (A/R) = {regrets[t, 2]}')
                logging.info(f'Instability (A/R) = {regrets[t, 3]}')

            observations = self.get_observations(x_OFU_intended, observations)
            observations = self.get_observations(x_OFU_uns, observations, zero_obs=True)

        self.save_results("LR-IR", regrets)
        return regrets

    def solve_CX_IR(self):

        # get initial observations
        observations = self.get_observations(self.initial_mask)

        logging.info(f'Running CX-IR')

        regrets = np.zeros((self.T, 4))

        x_prev = None

        for t in range(self.T):

            U = self.compute_center_contextual(observations)
            data_avg, counts = observations

            # obtain solution without capacity constraints
            x_OFU_intended = self.solve_for_contextual_ofu_allocation(U, counts, t,
                                                                      x_prev=x_prev, solve_with_capacity=False)
            x_OFU = x_OFU_intended.copy()

            # only some of the users obtain the items if there are too many requests for the same item
            for i in range(self.M):
                while np.sum(x_OFU[:, i]) > self.C[t, i]:
                    allocated_users = x_OFU[:, i] > 0.9
                    v = np.random.choice(np.where(allocated_users)[0])
                    x_OFU[v, i] = 0

            x_OFU_uns = x_OFU_intended - x_OFU

            p = np.zeros(self.M)
            regret_collection = self.calculate_regret(t, x_OFU, p, 0)
            regrets[t] = regret_collection

            if t % 10 == 0:
                logging.info(f'Iter {t}')
                logging.info(f'Social Welfare Regret = {regrets[t, 0]}')
                logging.info(f'Instability = {regrets[t, 1]}')
                logging.info(f'Social Welfare Regret (A/R) = {regrets[t, 2]}')
                logging.info(f'Instability (A/R) = {regrets[t, 3]}')

            observations = self.get_observations(x_OFU_intended, observations)
            observations = self.get_observations(x_OFU_uns, observations, zero_obs=True)

        self.save_results("CX-IR", regrets)
        return regrets



    def update_U(self, V, data_sum, counts, eps):
        wv = np.einsum('ui,ir->uir', counts, V)
        wv_t_v = np.einsum('uir,ik->urk', wv, V) + eps * np.eye(self.rank)
        wv_t_v_inv = np.linalg.inv(wv_t_v)
        v_s = np.einsum('ui,ir->ur', data_sum, V)
        U = np.einsum('ukr,ur->uk', wv_t_v_inv, v_s)
        return U

    def compute_center(self, observations):
        eps = 1e-4

        data_avg, counts = observations
        data_sum = data_avg * counts
        iter = 0

        V = np.random.rand(self.M, self.rank)
        U_prev = np.zeros((self.N, self.rank))
        while True:
            U = self.update_U(V, data_sum, counts, eps)
            V = self.update_U(U, data_sum.T, counts.T, eps)

            iter += 1
            if np.linalg.norm(U - U_prev) < np.maximum(self.N, self.M) * self.rank * 1e-5 or iter > 30:
                break
            U_prev = U.copy()

        return U, V

    def compute_center_contextual(self, observations):
        eps = 1e-4
        data_avg, counts = observations
        data_sum = data_avg * counts
        return self.update_U(self.V_true, data_sum, counts, eps)

    def solve_for_ofu_allocation(self, U, V, counts, t, x_prev=None, solve_with_capacity=True):
        beta1 = 8 * (self.eta ** 2) * (self.N + self.M + 1) * self.rank * \
                np.log(9 * self.T ** 2)
        beta2 = 2 * t / self.T * self.N * self.M \
                * (8 + np.sqrt(8 * (self.eta ** 2) * np.log(4 * np.sqrt(self.N * self.M) * (t+1) ** 2 * self.T)))

        beta = beta1 + beta2
        beta = beta * 1e-3

        alpha_UV = 1e-3
        Theta_LS = U @ V.T
        x = np.ones((self.N, self.M))
        def opt_UV(U, V, x):

            grad_U = x @ V
            U_hat = U + alpha_UV * grad_U
            U = conf_projection_u(U_hat, V, Theta_LS, counts, beta)

            #print(np.sum(counts * (U_hat @ V.T - Theta_LS) ** 2), np.sum(counts * (U @ V.T - Theta_LS) ** 2), beta)

            grad_V = x.T @ U
            V_hat = V + alpha_UV * grad_V
            V = conf_projection_v(V_hat, U, Theta_LS, counts, beta)

            return U, V

        if solve_with_capacity:
            U_prev = np.zeros_like(U)
            for i in range(2):
                for v in range(8):
                    if np.linalg.norm(U - U_prev) < 1e-6:
                        continue
                    U_prev = U.copy()
                    U, V = opt_UV(U, V, x)
                    # print(v, np.linalg.norm(U - U_prev))
                    if np.linalg.norm(U - U_prev) < 1e-6:
                        break

                R = U @ V.T
                x = self.solver.solve_system(R, self.C[t], self.D[t], x_prev=x_prev)

            # print(f"Mean delta for allocation: {np.mean(np.abs((R - Theta_LS) * x))}")
            # print(f"Delta L2: {np.linalg.norm(R - Theta_LS, ord='fro')}")
            # print(f"CI valid: {np.linalg.norm(Theta_LS - self.R_true, ord='fro'), np.sqrt(beta)}")
            # print()

        else:
            for v in range(10):
                U_prev = U.copy()
                U, V = opt_UV(U, V, x)
                if np.linalg.norm(U - U_prev) < 1e-6:
                    break
            R = U @ V.T
            x = self.solve_without_cap(R, t)

        return x

    def solve_for_contextual_ofu_allocation(self, U, counts, t, x_prev=None, solve_with_capacity=True):
        rho1 = 8 * (self.eta ** 2) * self.rank * np.log(3 * self.N * self.T ** 2)
        rho2 = 2 * t * np.sqrt(self.M) / self.T * ( 8 + np.sqrt(8 * (self.eta ** 2)
                                                              * np.log(4 * self.N * self.M * (t+1)**2 * self.T)))

        rho = rho1 + rho2
        rho = rho * 1e-2

        Theta_LS = U @ self.V_true.T

        x = np.ones((self.N, self.M))

        def opt_U(U, x):
            alpha_UV = 1e-2

            grad_U = x @ self.V_true
            U_hat = U + alpha_UV * grad_U
            U = conf_projection_u_inf(U_hat, self.V_true, Theta_LS, counts, rho)

            return U

        if solve_with_capacity:
            for i in range(10):
                U_prev = U.copy()
                U = opt_U(U, x)
                if np.linalg.norm(U - U_prev) < 1e-6:
                    break

                R = U @ self.V_true.T

            x = self.solver.solve_system(R, self.C[t], self.D[t], x_prev=x_prev)

            # print(f"Mean delta for allocation: {np.mean(np.abs((R - Theta_LS) * x))}")
            # print(f"Delta L2: {np.linalg.norm(R - Theta_LS, ord='fro')}")
            # print(f"CI valid: {np.max(np.linalg.norm(Theta_LS - self.R_true, axis=1)), np.sqrt(rho)}")
            # print()

        else:
            U = opt_U(U, x)
            R = U @ self.V_true.T
            x = self.solve_without_cap(R, t)

        return x

    def get_observations(self, X, history=None, zero_obs=False):
        gamma = 0.03
        if history is None:
            data_avg = np.zeros((self.N, self.M))
            data_count = np.zeros((self.N, self.M)) + gamma
        else:
            data_avg, data_count = history
        data_sum = data_avg * data_count
        if not zero_obs:
            new_data = self.R_true + self.eta * np.random.randn(self.N, self.M)
            data_sum = data_sum + X * new_data
        data_count += X
        data_avg = data_sum / data_count
        return data_avg, data_count


    def save_results(self, algo, regrets):
        np.save(f"{self.exp_save_path}/{algo}_regrets", regrets)
