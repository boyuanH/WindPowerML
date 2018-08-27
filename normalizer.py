# -*- coding: utf-8 -*-

import numpy as np


class StdGaussNormalizer:
    def __init__(self, dim, binwise=False):
        self.dim = dim
        self.mean = np.zeros(self.dim).astype(np.float32) if binwise else 0.0
        self.squared_mean = np.zeros(self.dim).astype(np.float32) if binwise else 0.0
        self.std = np.ones(self.dim).astype(np.float32) if binwise else 1.0
        self.data_count = 0
        self.binwise = binwise

    def update(self, x):
        pre_mean = self.mean
        pre_squared_mean = self.squared_mean
        adding_count = len(x) if self.binwise else x.size
        adding_mean = np.mean(x, axis=0) if self.binwise else np.mean(x)
        adding_squared_mean = np.mean(np.power(x, 2.0), axis=0) if self.binwise else np.mean(np.power(x, 2.0))
        self.data_count += adding_count
        alpha = adding_count / self.data_count
        self.mean = pre_mean + alpha * (adding_mean - pre_mean)
        self.squared_mean = pre_squared_mean + alpha * (adding_squared_mean - pre_squared_mean)
        self.std = np.sqrt(self.squared_mean - np.power(self.mean, 2.0))

    def __call__(self, x):
        return (x - self.mean) / self.std