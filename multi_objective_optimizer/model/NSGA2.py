import os
from copy import deepcopy
import random
import logging

import numpy as np

import multi_objective_optimizer.model.operator
from multi_objective_optimizer.model.settings import *

logger = logging.getLogger('info_logger')


class NSGA2(object):
    def __init__(self, npop, nc, eval_func, constraint_func,
                 init_mode='gaussian', init_range_lower=0, init_range_upper=1,
                 boundary_lower=0, boundary_upper=1,
                 selection_mode='tournament',
                 eta_d=2, tournament_size=2, mutation_probability=0.05):
        """
        GA class.
        :param npop: number of population.
        :param nc: number of children.
        :param eval_func: evaluation function to be optimized.
                          Note: return of eval_func must be matrix of row: individuals, col: corresponding objectives
        :param constraint_func: constraint function to be evaluated.
                          Note: return of eval_func must be a vector of dimension (individuals, )
        :param init_mode: initialization mode ('uniform' or 'gaussian').
        :param init_range_lower: range of init population (lower): only for uniform initialization
        :param init_range_upper: range of init population (upper): only for uniform initialization
        :param boundary_lower: boundary of variables (lower)
        :param boundary_upper: boundary of variables (upper)
        :param selection_mode: selection mode ('tournament' or 'random').
        :param eta_d: eta_d for SBX which controls range of children creation.
        :param tournament_size: number of tournament participants in selection.
        :param mutation_probability: probability for mutation.
        """
        # 集団など，GA内変数
        self.npop = npop
        self.nc = nc
        self.selection_mode = selection_mode
        self.eta_d = eta_d
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.first_group = np.zeros((self.npop, n))
        self.now_gen = 0
        self.R = []  # 旧集団と新集団の結合したもの
        self.P = []  # 旧集団
        self.Q = []  # 新集団
        self.F = [[] for v in range(10000)]  # Rをランクごとに分けて格納したもの
        # 各個体には，後ろに[評価関数値1-m,被ドミナント数，ランク，混雑度, [その個体にdominateされている個体番号群(Sp)], ペナルティを満たすかどうか]が付与されていることに注意
        # evaluation functions
        self.eval_func = eval_func
        self.constraint_func = constraint_func  # amount of constraint violation
        # ranges and boundary
        self.init_mode = init_mode
        self.init_range_lower = init_range_lower
        self.init_range_upper = init_range_upper
        self.boundary_lower = boundary_lower
        self.boundary_upper = boundary_upper
        # create output folder
        os.mkdir(result_dir)

    # 初期集団の生成と評価値の計算を行う
    def neutralization(self):
        for i in range(0, self.npop):
            if self.init_mode == 'gaussian':
                self.P.append(list(np.random.normal(size=n)))  # neutralization
            elif self.init_mode == 'uniform':
                self.P.append(list(self.init_range_lower + (self.init_range_upper - self.init_range_lower) * np.random.rand(n)))  # neutralization
            else:
                logger.error('unknown init mode.')
                exit(-1)
        self.now_gen = -1
        with open('Generation.csv', 'w') as f:
            print(str(self.now_gen + 1), file=f)
        J = self.eval_func(np.array(self.P).copy())
        J_constraint = self.constraint_func(np.array(self.P).copy())
        # 情報の付与
        for i in range(len(self.P)):
            for m in range(nobj):
                self.P[i].append(J[i, m])
            self.P[i].append(0)
            self.P[i].append(0)
            self.P[i].append(0)
            self.P[i].append([])
            self.P[i].append(J_constraint[i])
        self.now_gen = 0

        # initial assignment
        self.R = deepcopy(self.P)
        self.fast_NS()
        self.P = []
        i = 0
        while self.F[i]:
            self.crowding_distance_assignment(i)
            self.P = self.P + deepcopy(self.F[i])
            i += 1
        return

    @staticmethod
    def comparison(p, q):
        """
        return if p is dominant or not
        """
        flag = True
        # if both is infeasible
        if p[-1] > 0 and q[-1] > 0:
            if p[-1] > q[-1]:
                flag = False
        # if only p is infeasible
        elif p[-1] > 0 and q[-1] == 0:
            flag = False
        # if only q is infeasible
        elif p[-1] == 0 and q[-1] > 0:
            pass
        # if both are feasible
        else:
            for m in reversed(range(nobj)):
                if p[-1*(m+5+1)] > q[-1*(m+5+1)]:
                    flag = False
        return flag

    def fast_NS(self):
        self.F = [[] for v in range(10000)]
        for i in range(len(self.R)):
            self.R[i][-2] = [] # Sp初期化
            self.R[i][-5] = 0  # 被ドミナント数初期化
            for j in range(len(self.R)):
                if self.comparison(self.R[i],self.R[j]):
                    self.R[i][-2].append(j)
                elif self.comparison(self.R[j],self.R[i]):
                    self.R[i][-5] += 1
            if self.R[i][-5] == 0:
                self.R[i][-4] = 0
                self.F[0].append(deepcopy(self.R[i]))
        i = 0
        while self.F[i]:
            Q = []
            for p in self.F[i]:
                for j in p[-2]:
                    self.R[j][-5] -= 1
                    if self.R[j][-5] == 0:
                        self.R[j][-4] = i+1
                        Q.append(deepcopy(self.R[j]))
            i += 1
            self.F[i] = deepcopy(Q)

    def crowding_distance_assignment(self, i):
        l = len(self.F[i])
        for j in range(0, l):
            self.F[i][j][-3] = 0  # 距離情報の初期化
        for m in reversed(range(nobj)):  # 各目的関数値について
            self.F[i] = sorted(self.F[i], key=lambda p: p[-1*(m+5+1)])  # 目的関数の小さい順にソート
            fmax = self.F[i][l-1][-1*(m+5+1)]
            fmin = self.F[i][0][-1*(m+5+1)]
            self.F[i][0][-3] = INF
            self.F[i][l-1][-3] = INF
            for j in range(1, l-1, 1):
                self.F[i][j][-3] = self.F[i][j][-3] + (self.F[i][j+1][-1*(m+5+1)] - self.F[i][j-1][-1*(m+5+1)]) / (fmax - fmin + eps)

    def step(self):
        self.Q = self.make_new_pop()
        J = self.eval_func(np.array(self.Q).copy())
        if np.any(np.isnan(J)):
            print('nan is detected. skip.')
            with open('log.txt', 'a') as f:
                print('nan is detected. skip.', file=f)
            return
        J_constraint = self.constraint_func(np.array(self.Q).copy())
        # 情報の付与
        for i in range(len(self.Q)):
            for m in range(nobj):
                self.Q[i].append(J[i, m])
            self.Q[i].append(0)
            self.Q[i].append(0)
            self.Q[i].append(0)
            self.Q[i].append([])
            self.Q[i].append(J_constraint[i])
        self.R = deepcopy(self.P + self.Q)
        self.fast_NS()
        i = 0
        while self.F[i]:
            self.crowding_distance_assignment(i)
            i += 1
        self.P = []
        i = 0
        while len(self.P)+len(self.F[i]) <= self.npop:
            self.P = deepcopy(self.P + self.F[i])
            i += 1
        self.F[i] = sorted(self.F[i], key=lambda p: p[-3], reverse=True)  # 距離の大きい順にソート
        self.P = deepcopy(self.P + self.F[i][0:(self.npop-len(self.P))])
        self.now_gen += 1

    def selection_tournament(self, num_ind):

        def select_one_from_two_for_tournament(ind1, ind2):
            if self.comparison(ind1, ind2):
                return deepcopy(ind1)
            elif self.comparison(ind2, ind1):
                return deepcopy(ind2)
            if ind1[-3] > ind2[-3]:
                return deepcopy(ind1)
            else:
                return deepcopy(ind2)

        chosen_inds = []
        for i in range(num_ind):
            participants = [random.choice(self.P) for _ in range(self.tournament_size)]
            survived_inds = []
            survived_inds_old = deepcopy(participants)
            while len(survived_inds_old) > 1:
                survived_inds = []
                for k in range(0, len(survived_inds_old) - 1, 2):
                    survived_inds.append(select_one_from_two_for_tournament(survived_inds_old[k], survived_inds_old[k+1]))
                if len(survived_inds_old) % 2 == 1:  # あまりとのトーナメント
                    survived_inds_last = survived_inds[-1]
                    survived_inds.pop()
                    survived_inds.append(select_one_from_two_for_tournament(survived_inds_last, survived_inds_old[-1]))
                survived_inds_old = deepcopy(survived_inds)
            chosen_inds.append(deepcopy(survived_inds[0]))
        return chosen_inds

    def make_new_pop(self):  # SBX
        children = []
        while len(children) < self.nc:
            if self.selection_mode == 'tournament':
                parents = self.selection_tournament(2)
            elif self.selection_mode == 'random':
                parents = [random.choice(self.P) for _ in range(2)]
            else:
                logger.error('unknown selection mode.')
                exit(-1)
            parent1 = deepcopy(parents[0][:n])
            parent2 = deepcopy(parents[1][:n])
            child1, child2 = multi_objective_optimizer.model.operator.crossover_sbx(parent1, parent2, self.eta_d)
            child1 = multi_objective_optimizer.model.operator.mutation_gaussian(child1, self.mutation_probability)
            child2 = multi_objective_optimizer.model.operator.mutation_gaussian(child2, self.mutation_probability)
            child1 = np.where(child1 > self.boundary_upper, self.boundary_upper, child1)
            child1 = np.where(child1 < self.boundary_lower, self.boundary_lower, child1)
            child2 = np.where(child2 > self.boundary_upper, self.boundary_upper, child2)
            child2 = np.where(child2 < self.boundary_lower, self.boundary_lower, child2)
            children.append(deepcopy(list(child1)))
            children.append(deepcopy(list(child2)))
        return children
