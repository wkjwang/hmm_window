# -*- coding: utf-8 -*-
# Created by Kaijun WANG in May 2024
# Some codes in make_lagged_data() are created by Amal in Jun 2016.

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import lagmat2ds as lagmat2ds
from statsmodels.tools.tools import add_constant as add_constant
from statsmodels.regression.linear_model import OLS as OLS
# from sklearn.metrics import r2_score


def findzeros_variable(data):
	# count zeros in data, return their ratio
    nd = data.shape
    nzero = 0
    if len(nd) > 1:
        for i in range(nd[0]):
            if data[i, 0] == 0:
                nzero += 1
    else:
        for i in range(nd[0]):
            if data[i] == 0:
                nzero += 1
    nzero = nzero / nd[0]
    return nzero


def make_lagged_data(x, lag, addconst):
    x = np.asarray(x)
    # min data quantity to form time lag, and for degree of freedom in F_test
    num_lag = 2 * lag + int(addconst) + 1  # the best is 3lag, which is checked in calling function
    if x.shape[0] < num_lag:
        raise ValueError("Mimimum {0} observations are needed for lag {1}".format(num_lag, lag))
    # create lagmat from 0 to lag for 1st column/variable
    # create lagmat from dropex to lag for 2nd column/variable
    dta = lagmat2ds(x, lag, trim='both', dropex=1)
    # lag-variable data, without original variables
    if addconst:
        d_lag_own = add_constant(dta[:, 1:(lag + 1)], prepend=False, has_constant='add')
        d_lag_joint = add_constant(dta[:, 1:], prepend=False, has_constant='add')
    else:
        d_lag_own = dta[:, 1:(lag + 1)]
        d_lag_joint = dta[:, 1:]
    return dta[:, 0], d_lag_own, d_lag_joint


def r2_adjust(d_target, data_x, param):
    nrow, ncol = data_x.shape
    ncol = ncol - 1
    id = np.abs(param) < 0.02  # prevent residual drop with small parameters
    id[ncol] = False
    k = ncol - sum(id)
    param[id] = 0
    pred = data_x @ param
    ss_error = np.sum( (d_target - pred) ** 2)
    ss_total = np.sum( (d_target - np.average(d_target)) ** 2)
    # r2 = r2_score(d_target, pred)
    if ss_total < 0.0001:  # variation 1%; 0.001^2=1e-6
        ss_total = 0.0001
        if ss_error < 0.0001:
            ss_error = 0.0001
    r2 = 1 - ss_error/ss_total
    if np.isnan(r2) or r2 < 0:
        r2_adj = 0
    else:
        denominator = nrow - k - 1
        if denominator <= 0:
            r2_adj = r2
        else:
            r2_adj = 1 - (1 - r2) * (nrow - 1) / denominator
    if r2_adj > 1:
        r2_adj = 1
    return r2, r2_adj, ss_error


def regression_lag(x, lag, addconst=True):
    # response/target variable d_target
    d_target, d_lag_own, d_lag_joint = make_lagged_data(x, lag, addconst)
    # OLS: Fit a linear model using Ordinary Least Squares
    res2down = OLS(d_target, d_lag_own).fit()
    res2djoint = OLS(d_target, d_lag_joint).fit()
    r2, r2_adj_own, sse = r2_adjust(d_target, d_lag_own, res2down.params)
    r2, r2_adj_joint, sse = r2_adjust(d_target, d_lag_joint, res2djoint.params)
    return r2_adj_own, r2_adj_joint

'''

'''