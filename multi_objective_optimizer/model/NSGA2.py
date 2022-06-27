import os
from copy import deepcopy
import random

import numpy as np

from multi_objective_optimizer.model.settings import *


class NSGA2(object):
    def __init__(self, npop, nc, eval_func, constraint_func, init_range_lower=0, init_range_upper=1,
                 boundary_lower=0, boundary_upper=1, crossover_probability=0.9, eta_d=2):
        """
        GA class.
        :param npop: number of population.
        :param nc: number of children.
        :param eval_func: evaluation function to be optimized.
                          Note: return of eval_func must be matrix of row: individuals, col: corresponding objectives
        :param constraint_func: constraint function to be evaluated.
                          Note: return of eval_func must be a vector of dimension (individuals, )
        :param init_range_lower: range of init population (lower)
        :param init_range_upper: range of init population (upper)
        :param boundary_lower: boundary of variables (lower)
        :param boundary_upper: boundary of variables (upper)
        :param crossover_probability: crossover probability for SBX.
        :param eta_d: eta_d for SBX.
        """
        # 集団など，GA内変数
        self.npop = npop
        self.nc = nc
        self.crossover_probability = crossover_probability
        self.eta_d = eta_d
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
        self.init_range_lower = init_range_lower
        self.init_range_upper = init_range_upper
        self.boundary_lower = boundary_lower
        self.boundary_upper = boundary_upper
        # create output folder
        os.mkdir(result_dir)

    # 初期集団の生成と評価値の計算を行う
    def neutralization(self):
        for i in range(0, self.npop):
            # self.P.append(list(np.random.normal(size=n)))  # neutralization
            self.P.append(list(self.init_range_lower + (self.init_range_upper - self.init_range_lower) * np.random.rand(n)))  # neutralization
        self.now_gen = 99999
        with open('Generation.csv', 'w') as f:
            print(str(self.now_gen), file=f)
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
        self.now_gen += 1
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
        self.P = []
        i = 0
        while len(self.P)+len(self.F[i]) <= self.npop:
            self.P = deepcopy(self.P + self.F[i])
            i += 1
        self.crowding_distance_assignment(i)
        self.F[i] = sorted(self.F[i], key=lambda p: p[-3], reverse=True)  # 距離の大きい順にソート
        self.P = deepcopy(self.P + self.F[i][0:(self.npop-len(self.P))])

    def make_new_pop(self):  # SBX
        children = []
        population = deepcopy(self.P)
        population = np.array([l[:n] for l in population])
        while len(children) < self.nc:
            idx = random.sample(range(0, len(self.P), 1), k=2)
            child1 = []
            child2 = []
            parents = population[idx, :].copy()
            for j in range(n):
                if np.random.rand() < self.crossover_probability:
                    u = random.uniform(0, 1)
                    if u <= 0.5:
                        beta = (2*u)**(1.0 / (self.eta_d+1))
                    else:
                        beta = (1.0 / (2*(1-u)))**(1.0 / (self.eta_d+1))
                    child1.append(0.5*((1+beta)*parents[0, j] + (1-beta)*parents[1, j]))
                    child2.append(0.5*((1-beta)*parents[0, j] + (1+beta)*parents[1, j]))
                else:
                    child1.append(parents[0, j])
                    child2.append(parents[1, j])
            child1 = np.array(child1)
            child1 = np.where(child1 > self.boundary_upper, self.boundary_upper, child1)
            child1 = np.where(child1 < self.boundary_lower, self.boundary_lower, child1)
            child2 = np.array(child2)
            child2 = np.where(child2 > self.boundary_upper, self.boundary_upper, child2)
            child2 = np.where(child2 < self.boundary_lower, self.boundary_lower, child2)
            children.append(deepcopy(list(child1)))
            children.append(deepcopy(list(child2)))
        return children
