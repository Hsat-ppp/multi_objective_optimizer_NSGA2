"""
dd-CMA-ES optimizer.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np

from single_objective_optimizer.model.common_settings import *
from single_objective_optimizer.model import optimizer

approximated_E_normal = (np.sqrt(n) * (1.0 - (1.0 / (4.0 * n)) + (1.0 / (21.0 * n * n))))


class DDCMAES(optimizer.OPTIMIZER):
    """
    dd-CMA-ES optimizer class
    """
    def __init__(self, obj_func, population_size=int(4+np.floor(3*np.log(n))),
                 mu=None, mean=None, step_size=1.00,
                 cov_mat_coef=1.0, dd_mat_coef=1.0, t_eig=None,
                 plot_evolution_or_not=False):
        """
        init function where all parameters are initialized.
        :param obj_func: objective function to be minimized (input: data (pop_size*n mat), output: value (pop_size vec))
        :param population_size: population size (default value will be set)
        :param mu: up to mu-th population will be used for update with positive weight coefficient
        :param mean: distribution mean (if None, be set to np.zeros(n))
        :param step_size: step size (if None, be set to 1.00)
        :param cov_mat_coef: represent range of initial variation. That is, initial C = cov_mat_coef * I
        :param dd_mat_coef: represent range of initial variation. That is, initial D = dd_mat_coef * I
        :param t_eig: generation interval for eigen value decomposition of cov matrix
        :param plot_evolution_or_not: if you want to plot evolution, please set True (only valid when n=2)
        """
        super().__init__(obj_func, plot_evolution_or_not)

        # compute preliminary parameters
        self.population_size = population_size
        self.weight_pre = np.array([np.log((self.population_size + 1.0) / 2.0) - np.log(i)
                                    for i in range(1, self.population_size+1, 1)])
        if mu is None:
            self.mu = np.floor(self.population_size / 2.0).astype(np.int64)
        else:
            self.mu = mu
        self.mu_eff = (np.sum(self.weight_pre[:self.mu])**2) / (np.sum(self.weight_pre[:self.mu]**2))
        self.mu_eff_minus = (np.sum(self.weight_pre[self.mu:])**2) / (np.sum(self.weight_pre[self.mu:]**2))
        self.c_m = 1.0

        # step_size controller
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * np.max([0, np.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0]) + self.c_sigma

        # compute covariance matrix adaption coefficients
        m = n * (n + 1) / 2.0
        self.c_1 = 1.0 / (2.0 * (m / n + 1.0) * ((n + 1) ** (3.0 / 4.0)) + self.mu_eff / 2.0)
        m = n
        self.c_1_D = 1.0 / (2.0 * (m / n + 1.0) * ((n + 1) ** (3.0 / 4.0)) + self.mu_eff / 2.0)
        mu_dash = self.mu_eff + 1.0 / self.mu_eff - 2.0 + (1.0 / 2.0) * (
                    self.population_size / (self.population_size + 5.0))
        self.c_mu = np.min([mu_dash * self.c_1, 1.0 - self.c_1])
        self.c_mu_D = np.min([mu_dash * self.c_1_D, 1.0 - self.c_1_D])
        self.c_c = np.sqrt(self.mu_eff * self.c_1) / 2.0
        self.c_c_D = np.sqrt(self.mu_eff * self.c_1_D) / 2.0

        # compute weight coefficients
        self.sum_of_weight_pre_plus = np.sum(np.abs(self.weight_pre[self.weight_pre >= 0]))
        self.sum_of_weight_pre_minus = np.sum(np.abs(self.weight_pre[self.weight_pre < 0]))
        self.weight = np.zeros_like(self.weight_pre)
        r = self.c_1 / self.c_mu
        for i in range(len(self.weight)):
            if self.weight_pre[i] >= 0:
                self.weight[i] = self.weight_pre[i] * 1.0 / self.sum_of_weight_pre_plus
            else:
                self.weight[i] = self.weight_pre[i] * np.min(
                    [1.0 + r, 1.0 + 2.0 * self.mu_eff_minus / (self.mu_eff + 2.0)]) / self.sum_of_weight_pre_minus
        self.weight_D = np.zeros_like(self.weight_pre)
        r = self.c_1_D / self.c_mu_D
        for i in range(len(self.weight_D)):
            if self.weight_pre[i] >= 0:
                self.weight_D[i] = self.weight_pre[i] * 1.0 / self.sum_of_weight_pre_plus
            else:
                self.weight_D[i] = self.weight_pre[i] * np.min(
                    [1.0 + r, 1.0 + 2.0 * self.mu_eff_minus / (self.mu_eff + 2.0)]) / self.sum_of_weight_pre_minus

        # eigen value decomposition interval
        if t_eig is None:
            self.t_eig = int(np.max([1.0, np.floor((10.0 * n * (self.c_1 + self.c_mu))**(-1))]))
        else:
            self.t_eig = t_eig

        # threshold for damping factor
        self.beta_thresh = 2.0

        # cov matrix and eigen matrix
        self.cov_matrix = cov_mat_coef * np.eye(n)  # default: C=I
        self.eigen_values = None
        self.eigen_vec_matrix = None
        self.eigen_val_matrix = None
        self.cov_mat_sqrt = None
        self.cov_mat_inv_sqrt = None
        self.eigen_value_decomposition()

        # active update matrix represented as Z, and its sum K
        self.z_matrix = np.zeros((n, n))
        self.k_matrix = np.zeros((n, n))

        # diagonal decoding matrix
        self.dd_matrix = dd_mat_coef * np.eye(n)  # default: D=I
        self.dd_matrix_inv = np.eye(n)
        for k in range(n):
            self.dd_matrix_inv[k, k] = 1.0 / self.dd_matrix[k, k]

        # evolution paths
        self.evol_path_sigma = np.zeros(n)
        self.evol_path_c = np.zeros(n)
        self.evol_path_c_D = np.zeros(n)

        # normalization factors
        self.gamma_c = 0.0
        self.gamma_c_D = 0.0
        self.gamma_sigma = 0.0

        # damping factor
        self.beta = 1.0

        # x, y, z and evaluation by obj_func
        self.x = np.zeros((self.population_size, n))
        self.y = np.zeros((self.population_size, n))
        self.z = np.zeros((self.population_size, n))
        self.y_rescaled = np.zeros((self.population_size, n))  # rescaled vec to controller positive definiteness
        self.z_rescaled = np.zeros((self.population_size, n))  # rescaled vec to controller positive definiteness
        self.evaluation_vec = np.zeros(self.population_size)

        # mean and step size
        if mean is None:
            self.mean = np.zeros(n)
        else:
            self.mean = copy.deepcopy(mean)
        self.step_size = step_size

    def proceed_generation(self):
        """
        proceed generation calling all operation
        :return:
        """
        self.sample_new_population()
        self.evaluation_and_sort()
        if self.plot_evolution_or_not and n == 2:
            self.plot_evolution()
        self.selection_and_recombination()
        self.step_size_control()
        self.evolution_path_update()
        self.rescale_y_and_z()
        self.store_update_for_cov_matrix()
        self.dd_matrix_update()
        self.generation += 1
        if (self.generation % self.t_eig) == 0:
            self.cov_matrix_adaption()
            self.matrix_reparametrization()
            self.eigen_value_decomposition()
            self.damping_factor_update()
        self.dd_matrix_inversion()

    def proceed_generation_active_cmaes(self):
        """
        Proceed generation calling all operation but keeping D=I, which results in active CMA-ES.
        :return:
        """
        self.sample_new_population()
        self.evaluation_and_sort()
        if self.plot_evolution_or_not and n == 2:
            self.plot_evolution()
        self.selection_and_recombination()
        self.step_size_control()
        self.evolution_path_update()
        self.rescale_y_and_z()
        self.store_update_for_cov_matrix()
        self.generation += 1
        self.cov_matrix_adaption()
        self.eigen_value_decomposition()

    def proceed_generation_sep_cmaes(self):
        """
        Proceed generation calling all operation but keeping C=I, which results in separable CMA-ES.
        :return:
        """
        self.sample_new_population()
        self.evaluation_and_sort()
        if self.plot_evolution_or_not and n == 2:
            self.plot_evolution()
        self.selection_and_recombination()
        self.step_size_control()
        self.evolution_path_update()
        self.rescale_y_and_z()
        self.dd_matrix_update()
        self.generation += 1
        self.dd_matrix_inversion()

    def sample_new_population(self):
        """
        Sample new population from N(mean, (step_size**2) * DCD), C is cov matrix, D is diagonal decoding matrix
        :return:
        """
        # generate sampling points
        self.z = np.random.normal(loc=0.0, scale=1.0, size=(self.population_size, n))
        self.z = self.z.T  # transpose to compute for each individuals
        # dd-cmaesの枠組みだと，ここをEL^(1/2)zにしてしまうと性能が著しく損なわれる．
        # おそらく，後の演算全てでy=sqrt(C)zという記法を採用しているため．ここが食い違うと性能が悪化する？
        self.y = self.cov_mat_sqrt @ self.z
        self.z = self.z.T  # recover transpose
        self.x = self.mean + self.step_size * (self.dd_matrix @ self.y).T  # dd-matrix is introduced
        self.y = self.y.T  # recover transpose

    def evaluation_and_sort(self):
        """
        Evaluate individuals x and sort x, y, z in increasing order.
        As a result, f(x[1]) < f(x[2]) < ... < f(x[population_size]).
        :return:
        """
        self.evaluation_vec = self.obj_func(self.x)
        self.num_of_evaluation += self.x.shape[0]
        ranks = np.argsort(self.evaluation_vec)
        # sort
        self.evaluation_vec = self.evaluation_vec[ranks]
        self.x = self.x[ranks, :]
        self.y = self.y[ranks, :]
        self.z = self.z[ranks, :]
        # elite preservation
        if self.best_eval > self.evaluation_vec[0]:
            self.best_eval = self.evaluation_vec[0]
            self.best_solution = copy.deepcopy(self.x[0, :])

    def selection_and_recombination(self):
        """
        Update mean vector, computing sum of w * (x - mean), using best mu individuals
        :return:
        """
        x_diff_sum_w = np.zeros(n)
        for i in range(self.mu):
            x_diff_sum_w += self.weight[i] * (self.x[i, :] - self.mean)
        self.mean = self.mean + self.c_m * x_diff_sum_w

    def step_size_control(self):
        """
        Update step size, computing evolution path sigma.
        :return:
        """
        z_sum_w = np.zeros(n)
        for i in range(self.mu):
            z_sum_w += self.weight[i] * self.z[i, :]
        self.evol_path_sigma = ((1.0 - self.c_sigma) * self.evol_path_sigma
                                + np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * z_sum_w)
        self.gamma_sigma = ((1.0 - self.c_sigma)**2) * self.gamma_sigma + self.c_sigma * (2.0 - self.c_sigma)
        self.step_size = (self.step_size
                          * np.exp((self.c_sigma / self.d_sigma)
                                   * ((np.linalg.norm(self.evol_path_sigma) / approximated_E_normal)
                                      - np.sqrt(self.gamma_sigma))))

    def evolution_path_update(self):
        """
        Update evolution path c and c_D. Also update normalization factors, gamma_c and gamma_c_D.
        :return:
        """
        h_sigma = 0
        if (np.linalg.norm(self.evol_path_sigma) ** 2) / self.gamma_sigma < (2.0 + (4.0 / (n + 1.0))) * n:
            h_sigma = 1.0
        y_sum_w_D = np.zeros(n)
        for i in range(self.mu):
            y_sum_w_D += self.weight[i] * (self.dd_matrix @ (self.y[i, :].reshape((n, 1)))).reshape(n)

        # update evol path c
        self.evol_path_c = (1.0 - self.c_c) * self.evol_path_c + h_sigma * np.sqrt(
            self.c_c * (2.0 - self.c_c) * self.mu_eff) * y_sum_w_D

        # update evol path c_D
        self.evol_path_c_D = (1.0 - self.c_c_D) * self.evol_path_c_D + h_sigma * np.sqrt(
            self.c_c_D * (2.0 - self.c_c_D) * self.mu_eff) * y_sum_w_D

        # update normalization params
        self.gamma_c = ((1.0 - self.c_c)**2) * self.gamma_c + h_sigma * self.c_c * (2.0 - self.c_c)
        self.gamma_c_D = ((1.0 - self.c_c_D)**2) * self.gamma_c_D + h_sigma * self.c_c_D * (2.0 - self.c_c_D)

    def rescale_y_and_z(self):
        """
        Rescale y and z, which makes it easy to controller positive definiteness of cov matrix.
        :return:
        """
        for i in range(self.population_size):
            if self.weight[i] >= 0:
                self.y_rescaled[i, :] = copy.deepcopy(self.y[i, :])
                self.z_rescaled[i, :] = copy.deepcopy(self.z[i, :])
            else:
                self.y_rescaled[i, :] = np.sqrt(n) * self.y[i, :] / np.linalg.norm(self.z[i, :])
                self.z_rescaled[i, :] = np.sqrt(n) * self.z[i, :] / np.linalg.norm(self.z[i, :])

    def store_update_for_cov_matrix(self):
        """
        Store update amount Z to K, which is later used to update cov matrix.
        :return:
        """
        rank_one_product = self.cov_mat_inv_sqrt @ self.dd_matrix_inv @ (self.evol_path_c.reshape((n, 1)))
        rank_one_update_value = rank_one_product @ rank_one_product.T
        rank_mu_update_value = np.zeros((n, n))
        for i in range(self.population_size):
            rank_mu_update_value += self.weight[i] * ((self.z_rescaled[i, :].reshape((n, 1)))
                                                      @ (self.z_rescaled[i, :].reshape((1, n)))
                                                      - np.eye(n))
        self.z_matrix = self.c_1 * (rank_one_update_value - self.gamma_c * np.eye(n)) + self.c_mu * rank_mu_update_value
        self.k_matrix += self.z_matrix

    def dd_matrix_update(self):
        """
        Update diagonal decoding matrix D.
        :return:
        """
        delta_D_vec = np.zeros(n)
        evol_path_product = self.cov_mat_inv_sqrt @ self.dd_matrix_inv @ (self.evol_path_c_D.reshape((n, 1)))
        evol_path_product = evol_path_product.reshape(n)
        rank_mu_update_values = np.zeros(n)
        for i in range(self.population_size):
            rank_mu_update_values += self.weight_D[i] * (self.z_rescaled[i, :]**2 - 1)

        for k in range(n):
            rank_one_update_value = evol_path_product[k]**2
            rank_mu_update_value = rank_mu_update_values[k]
            delta_D_vec[k] = self.c_1_D * (rank_one_update_value - self.gamma_c_D) + self.c_mu_D * rank_mu_update_value

        self.dd_matrix = self.dd_matrix @ np.diag(np.exp(delta_D_vec / (2.0 * self.beta)))

    def cov_matrix_adaption(self):
        """
        Update cov matrix, using stored update matrix K.
        :return:
        """
        # compute alpha
        eigen_values_of_k_matrix, _ = np.linalg.eig(self.k_matrix)
        eigen_values_of_k_matrix = eigen_values_of_k_matrix.real
        minimum_eigen_value_of_k_matrix = np.min(eigen_values_of_k_matrix)
        alpha = np.min([0.75 / np.abs(minimum_eigen_value_of_k_matrix), 1.0])
        # compute cov matrix
        self.cov_matrix = self.cov_mat_sqrt @ (np.eye(n) + alpha * self.k_matrix) @ self.cov_mat_sqrt
        # reset k matrix
        self.k_matrix = np.zeros((n, n))

    def matrix_reparametrization(self):
        """
        Re-parametrize C and D for computation stability.
        :return:
        """
        cov_matrix_diag_sqrt = np.diag(np.sqrt(np.diag(self.cov_matrix)))
        cov_matrix_diag_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(self.cov_matrix)))
        self.dd_matrix = self.dd_matrix @ cov_matrix_diag_sqrt
        self.cov_matrix = cov_matrix_diag_sqrt_inv @ self.cov_matrix @ cov_matrix_diag_sqrt_inv

    def eigen_value_decomposition(self):
        """
        Perform eigen value decomposition of cov matrix.
        Also compute sqrt of inverse of cov matrix for following computation.
        :return:
        """
        self.eigen_values, self.eigen_vec_matrix = np.linalg.eig(self.cov_matrix)
        self.eigen_values = self.eigen_values.real  # 複素数が混じる場合があるため，実数のみ取り出す (おそらく問題は生じない)
        self.eigen_vec_matrix = self.eigen_vec_matrix.real  # 複素数が混じる場合があるため，実数のみ取り出す (おそらく問題は生じない)
        self.eigen_val_matrix = np.diag(np.sqrt(self.eigen_values))  # note: defined as sqrt of eigen values as in original CMA-ES paper.
        self.cov_mat_sqrt = self.eigen_vec_matrix @ np.diag(np.sqrt(self.eigen_values)) @ self.eigen_vec_matrix.T
        self.cov_mat_inv_sqrt = self.eigen_vec_matrix @ np.diag(1.0 / np.sqrt(self.eigen_values)) @ self.eigen_vec_matrix.T

    def dd_matrix_inversion(self):
        """
        Compute inverse of D.
        :return:
        """
        for k in range(n):
            self.dd_matrix_inv[k, k] = 1.0 / self.dd_matrix[k, k]

    def damping_factor_update(self):
        """
        Update damping factor beta.
        :return:
        """
        maximum_eigen_value = np.max(self.eigen_values)
        minimum_eigen_value = np.min(self.eigen_values)
        self.beta = np.max([1.0, np.sqrt(maximum_eigen_value / minimum_eigen_value) - self.beta_thresh + 1.0])

    def plot_evolution(self):
        """
        Plot evolution of population on the contour color map of obj function.
        :return:
        """
        ax = plt.subplot()

        # obj_funcのコンター図
        ax.pcolormesh(self.X_PLOT, self.Y_PLOT, self.obj_func_profit, cmap='viridis', shading='auto')

        # Cの等確率線
        def gaussian_function(cx, cy):
            x = np.array([cx, cy])
            return np.exp(-0.5 * (x - self.mean).T @ np.linalg.inv(self.step_size * self.step_size * self.dd_matrix @ self.cov_matrix @ self.dd_matrix) @ (x - self.mean)) / np.sqrt(
                np.linalg.det(self.step_size * self.step_size * self.dd_matrix @ self.cov_matrix @ self.dd_matrix) * (2 * np.pi) ** n)

        C_profit = np.vectorize(gaussian_function)(self.X_PLOT, self.Y_PLOT)
        ax.contour(self.X_PLOT, self.Y_PLOT, C_profit, levels=[i for i in np.arange(0.01, 0.10, 0.01)],
                   colors=['yellow'], linewidths=[0.3], linestyles=['dashed'])

        # sample pointsの散布図，中心点，および最良解
        ax.scatter(self.x[:, 0], self.x[:, 1], color='black')
        ax.scatter(self.mean[0], self.mean[1], color='green')
        ax.scatter(self.best_solution[0], self.best_solution[1], color='red')

        # label
        plt.xlabel('x_1')
        plt.ylabel('x_2')

        ax.set_xlim(plot_range)
        ax.set_ylim(plot_range)
        ax.set_aspect('equal')

        plt.savefig(str(self.generation + 1).zfill(2) + '.png')

        plt.clf()
