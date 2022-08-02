import copy
import os
from copy import deepcopy
import random
import logging

import numpy as np

import multi_objective_optimizer.model.operator
from multi_objective_optimizer.model.settings import *

logger = logging.getLogger('info_logger')


class EAPES(object):
    def __init__(self, Fnpop, Inpop, nc, eval_func, constraint_func,
                 init_mode='gaussian', init_range_lower=0, init_range_upper=1,
                 boundary_lower=0, boundary_upper=1,
                 selection_mode='tournament',
                 eta_d=2, tournament_size=2, mutation_probability=0.05):
        """
        EAPES class.
        :param Fnpop: number of feasible population.
        :param Inpop: number of infeasible population.
        :param nc: number of children.
        :param eval_func: evaluation function to be optimized.
                          Note: return of eval_func must be matrix of row: individuals, col: corresponding objectives
        :param constraint_func: constraint function to be evaluated.
                          Note: return of eval_func must be a vector of dimension (individuals, )
        :param init_mode: initialization mode ('uniform' or 'gaussian').
        :param init_range_lower: range of init population (lower)
        :param init_range_upper: range of init population (upper)
        :param boundary_lower: boundary of variables (lower)
        :param boundary_upper: boundary of variables (upper)
        :param selection_mode: selection mode ('tournament' or 'random').
        :param eta_d: eta_d for SBX.
        :param tournament_size: number of tournament participants in selection.
        :param mutation_probability: probability for mutation.
        """
        # 集団など，GA内変数
        self.Fnpop = Fnpop
        self.Inpop = Inpop
        self.nc = nc
        self.selection_mode = selection_mode
        self.eta_d = eta_d
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.first_group = np.zeros((self.Fnpop + self.Inpop, n))
        self.now_gen = 0
        # F: feasible  I: infeasible
        self.P = []  # 旧集団
        self.FR = []  # 旧集団と新集団の結合したもの
        self.FP = []  # 旧集団
        self.IR = []  # 旧集団と新集団の結合したもの
        self.IP = []  # 旧集団
        self.Q = []  # 新集団
        self.FQ = []  # 新集団
        self.IQ = []  # 新集団
        self.FF = [[] for v in range(10000)]  # Rをランクごとに分けて格納したもの
        self.IF = [[] for v in range(10000)]  # Rをランクごとに分けて格納したもの
        self.Pareto = []
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
        for i in range(0, self.Fnpop + self.Inpop):
            if self.init_mode == 'gaussian':
                self.P.append(list(np.random.normal(size=n)))  # neutralization
            elif self.init_mode == 'uniform':
                self.P.append(list(
                    self.init_range_lower + (self.init_range_upper - self.init_range_lower) * np.random.rand(
                        n)))  # neutralization
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
        # feasibleを分類
        for i in range(len(self.P)):
            if self.P[i][-1] == 0:
                self.FP.append(deepcopy(self.P[i]))
        # infeasibleを分類
        for i in range(len(self.P)):
            if self.P[i][-1] > 0:
                # feasibleに支配されないものだけ保存
                flag = True
                for j in range(len(self.FP)):
                    if self.comparison_feasible(self.FP[j], self.P[i]):
                        flag = False
                        break
                if flag:
                    self.IP.append(deepcopy(self.P[i]))
        # initial assignment
        self.FR = deepcopy(self.FP)
        self.fast_NS_feasible()
        self.FP = []
        i = 0
        while self.FF[i]:
            self.crowding_distance_assignment_feasible(i)
            self.FP = self.FP + deepcopy(self.FF[i])
            i += 1
        # initial assignment
        self.IR = deepcopy(self.IP)
        self.fast_NS_infeasible()
        self.IP = []
        i = 0
        while self.IF[i]:
            self.crowding_distance_assignment_infeasible(i)
            self.IP = self.IP + deepcopy(self.IF[i])
            i += 1
        # FPとIPを統合
        self.P = deepcopy(self.FP + self.IP)
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
            flag = EAPES.comparison_infeasible(p, q)
        # if only p is infeasible
        elif p[-1] > 0 and q[-1] == 0:
            flag = False
        # if only q is infeasible
        elif p[-1] == 0 and q[-1] > 0:
            pass
        # if both are feasible
        else:
            flag = EAPES.comparison_feasible(p, q)
        return flag

    @staticmethod
    def comparison_feasible(p, q):
        """
        return if p is dominant or not
        """
        flag = True
        for m in reversed(range(nobj)):
            if p[-1 * (m + 5 + 1)] > q[-1 * (m + 5 + 1)]:
                flag = False
        return flag

    @staticmethod
    def comparison_infeasible(p, q):
        """
        return if p is dominant or not
        """
        flag = True
        for m in reversed(range(nobj)):
            # infeasibleについては最大化＝負の最小化 を考える
            if -1 * p[-1*(m+5+1)] > -1 * q[-1*(m+5+1)]:
                flag = False
        # 制約違反値についても比較
        if p[-1] > q[-1]:
            flag = False
        return flag

    def fast_NS_feasible(self):
        self.FF = [[] for v in range(10000)]
        for i in range(len(self.FR)):
            self.FR[i][-2] = [] # Sp初期化
            self.FR[i][-5] = 0  # 被ドミナント数初期化
            for j in range(len(self.FR)):
                if self.comparison_feasible(self.FR[i], self.FR[j]):
                    self.FR[i][-2].append(j)
                elif self.comparison_feasible(self.FR[j], self.FR[i]):
                    self.FR[i][-5] += 1
            if self.FR[i][-5] == 0:
                self.FR[i][-4] = 0
                self.FF[0].append(deepcopy(self.FR[i]))
        i = 0
        while self.FF[i]:
            FQ = []
            for p in self.FF[i]:
                for j in p[-2]:
                    self.FR[j][-5] -= 1
                    if self.FR[j][-5] == 0:
                        self.FR[j][-4] = i+1
                        FQ.append(deepcopy(self.FR[j]))
            i += 1
            self.FF[i] = deepcopy(FQ)

    def fast_NS_infeasible(self):
        self.IF = [[] for v in range(10000)]
        for i in range(len(self.IR)):
            self.IR[i][-2] = [] # Sp初期化
            self.IR[i][-5] = 0  # 被ドミナント数初期化
            for j in range(len(self.IR)):
                if self.comparison_infeasible(self.IR[i], self.IR[j]):
                    self.IR[i][-2].append(j)
                elif self.comparison_infeasible(self.IR[j], self.IR[i]):
                    self.IR[i][-5] += 1
            if self.IR[i][-5] == 0:
                self.IR[i][-4] = 0
                self.IF[0].append(deepcopy(self.IR[i]))
        i = 0
        while self.IF[i]:
            IQ = []
            for p in self.IF[i]:
                for j in p[-2]:
                    self.IR[j][-5] -= 1
                    if self.IR[j][-5] == 0:
                        self.IR[j][-4] = i+1
                        IQ.append(deepcopy(self.IR[j]))
            i += 1
            self.IF[i] = deepcopy(IQ)

    def crowding_distance_assignment_feasible(self, i):
        l = len(self.FF[i])
        for j in range(0, l):
            self.FF[i][j][-3] = 0  # 距離情報の初期化
        for m in reversed(range(nobj)):  # 各目的関数値について
            self.FF[i] = sorted(self.FF[i], key=lambda p: p[-1*(m+5+1)])  # 目的関数の小さい順にソート
            fmax = self.FF[i][l-1][-1*(m+5+1)]
            fmin = self.FF[i][0][-1*(m+5+1)]
            self.FF[i][0][-3] = INF
            self.FF[i][l-1][-3] = INF
            for j in range(1, l-1, 1):
                self.FF[i][j][-3] = self.FF[i][j][-3] + (self.FF[i][j+1][-1*(m+5+1)] - self.FF[i][j-1][-1*(m+5+1)]) / (fmax - fmin + eps)

    def crowding_distance_assignment_infeasible(self, i):
        l = len(self.IF[i])
        for j in range(0, l):
            self.IF[i][j][-3] = 0  # 距離情報の初期化
        for m in reversed(range(nobj)):  # 各目的関数値について
            self.IF[i] = sorted(self.IF[i], key=lambda p: p[-1*(m+5+1)])  # 目的関数の小さい順にソート
            fmax = self.IF[i][l-1][-1*(m+5+1)]
            fmin = self.IF[i][0][-1*(m+5+1)]
            self.IF[i][0][-3] = INF
            self.IF[i][l-1][-3] = INF
            for j in range(1, l-1, 1):
                self.IF[i][j][-3] = self.IF[i][j][-3] + (self.IF[i][j+1][-1*(m+5+1)] - self.IF[i][j-1][-1*(m+5+1)]) / (fmax - fmin + eps)
        # 違反量について
        self.IF[i] = sorted(self.IF[i], key=lambda p: p[-1])  # 目的関数の小さい順にソート
        fmax = self.IF[i][l - 1][-1]
        fmin = self.IF[i][0][-1]
        self.IF[i][0][-3] = INF
        self.IF[i][l - 1][-3] = INF
        for j in range(1, l - 1, 1):
            self.IF[i][j][-3] = self.IF[i][j][-3] + (
                        self.IF[i][j + 1][-1] - self.IF[i][j - 1][-1]) / (fmax - fmin + eps)

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
        # feasibleを分類
        self.FQ = []
        for i in range(len(self.Q)):
            if self.Q[i][-1] == 0:
                self.FQ.append(deepcopy(self.Q[i]))
        self.FR = deepcopy(self.FP + self.FQ)
        # infeasibleを分類
        self.IQ = []
        for i in range(len(self.Q)):
            if self.Q[i][-1] > 0:
                # feasibleに支配されないものだけ保存
                flag = True
                for j in range(len(self.FR)):
                    if self.comparison_feasible(self.FR[j], self.Q[i]):
                        flag = False
                        break
                if flag:
                    self.IQ.append(deepcopy(self.Q[i]))
        # IPを再分類
        IP_temp = []
        for i in range(len(self.IP)):
            # feasibleに支配されないものだけ保存
            flag = True
            for j in range(len(self.FR)):
                if self.comparison_feasible(self.FR[j], self.IP[i]):
                    flag = False
                    break
            if flag:
                IP_temp.append(deepcopy(self.IP[i]))
        self.IP = deepcopy(IP_temp)
        self.IR = deepcopy(self.IP + self.IQ)
        # preserve feasible solutions
        self.fast_NS_feasible()
        self.Pareto = deepcopy(self.FF[0])  # Pareto解の保存
        i = 0
        while self.FF[i]:
            self.crowding_distance_assignment_feasible(i)
            i += 1
        self.FP = []
        i = 0
        flag = True
        while len(self.FP)+len(self.FF[i]) <= self.Fnpop:
            self.FP = deepcopy(self.FP + self.FF[i])
            i += 1
            # 次のrankが存在しなければ，終了
            if not self.FF[i]:
                flag = False
                break
        if flag:
            self.FF[i] = sorted(self.FF[i], key=lambda p: p[-3], reverse=True)  # 距離の大きい順にソート
            self.FP = deepcopy(self.FP + self.FF[i][0:(self.Fnpop-len(self.FP))])
        # preserve infeasible solutions
        self.fast_NS_infeasible()
        i = 0
        while self.IF[i]:
            self.crowding_distance_assignment_infeasible(i)
            i += 1
        self.IP = []
        i = 0
        flag = True
        while len(self.IP) + len(self.IF[i]) <= self.Inpop:
            self.IP = deepcopy(self.IP + self.IF[i])
            i += 1
            # 次のrankが存在しなければ，終了
            if not self.IF[i]:
                flag = False
                break
        if flag:
            self.IF[i] = sorted(self.IF[i], key=lambda p: p[-3], reverse=True)  # 距離の大きい順にソート
            self.IP = deepcopy(self.IP + self.IF[i][0:(self.Inpop - len(self.IP))])
            self.now_gen += 1
        # Pareto解の統合
        self.P = deepcopy(self.FP + self.IP)

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
