"""
General optimizer class.
"""

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common_settings import *


class OPTIMIZER(object):
    """
    General optimizer class.
    """
    def __init__(self, obj_func, plot_evolution_or_not=False):
        """
        init function where all parameters are initialized.
        :param obj_func: objective function to be minimized (input: data (pop_size*n mat), output: value (pop_size vec))
        :param plot_evolution_or_not: if you want to plot evolution, please set True (only valid when n=2)
        """
        self.obj_func = obj_func

        # elite preservation
        self.best_eval = INF
        self.best_solution = np.zeros(n)

        # generation
        self.generation = 0
        self.num_of_evaluation = 0

        # plot settings
        self.plot_evolution_or_not = plot_evolution_or_not
        if self.plot_evolution_or_not and n == 2:
            self.x_plot = np.arange(plot_range[0], plot_range[1], width_for_plot)
            self.y_plot = np.arange(plot_range[0], plot_range[1], width_for_plot)
            self.X_PLOT, self.Y_PLOT = np.meshgrid(self.x_plot, self.y_plot)
            self.profit_points = np.array([self.X_PLOT.ravel(), self.Y_PLOT.ravel()]).T
            self.obj_func_profit = self.obj_func(self.profit_points).reshape((len(self.x_plot), len(self.y_plot)))
