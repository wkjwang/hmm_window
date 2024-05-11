# -*- coding: utf-8 -*-
# Created by Kaijun WANG in May 2024
# Some Granger_test codes, from F_statistic_bound method to find varying Granger causal relations
# of time series, are created by Amal in Jun 2016.

import numpy as np
from scipy import stats
from hmmlearn import hmm
from regression_withlag import *


def run_HMM(obs_data):
    # input: observation series, return: hidden-state series
    model = hmm.GaussianHMM(n_components=2, n_iter=1000, tol=0.0001)
    logprob_best = -10000
    states_best = None
    for i in range(0, 10):
        model.fit(obs_data.reshape(-1, 1))
        logprob, states = model.decode(obs_data.reshape(-1, 1), algorithm="viterbi")
        if logprob > logprob_best:
            logprob_best = logprob
            states_best = states
    return states_best


def counts2label(result, nsize, labels=[0, 0]):
    qc = np.zeros((nsize, 1), dtype=int)  # count time points where causality occurs
    qn = np.zeros((nsize, 1), dtype=int)  # count time points where causality disappears
    labels = np.array([labels], dtype=int)
    ns = labels.shape
    qout = np.zeros(nsize, dtype=int)
    for item in result:
        i, j = item[0], item[1]
        val = result[item]
        if val == 0:
            for k in range(i, j + 1):
                qn[k] += 1
        else:
            for k in range(i, j + 1):
                qc[k] += 1
    for i in range(0, nsize):
        if qc[i] > qn[i]:
                qout[i] = 1  # 1: causality appears
    if ns[1] > 5:
        q_all = np.hstack((qc, qn))
        q_all = np.hstack((q_all, labels.T))
        return qout, q_all
    return qout


def counts2score(result, nsize, labels=[0, 0]):
    qc = np.zeros((nsize, 1), dtype=float)  # count time points where causality occurs
    qn = np.zeros((nsize, 1), dtype=float)  # count time points where causality disappears
    score = np.zeros((nsize, 1), dtype=float)
    qout = np.zeros(nsize, dtype=int)
    labels = np.array([labels], dtype=int)
    ns = labels.shape
    smax = 0.0
    for item in result:
        i, j = item[0], item[1]
        val = result[item]
        if val == 0:
            for k in range(i, j + 1):
                qn[k] += 1
        else:
            for k in range(i, j + 1):
                qc[k] += 1
    for i in range(0, nsize):
        val = int(qc[i] + qn[i])
        if val > 0:
            val = int(qc[i])/val
            score[i] = val  # causal score
            if val > smax:
                smax = val
    smax = 0.9 * smax  # theshold for causality occurs
    for i in range(0, nsize):
        if score[i] > smax:
            qout[i] = 1
    if ns[1] > 5:
        q_all = np.hstack((qc, qn))
        q_all = np.hstack((q_all, score))
        return qout, q_all
    return qout


def accuracy_rate(result, causal_label):
    nsize = len(causal_label)
    acy = 0
    for i in range(0, nsize):
        if result[i] == causal_label[i]:
            acy += 1
    acy /= nsize
    return acy


def count_consecutive_values(sequence):
    # count consecutive values for each value
    # e.g. sequence = [6, 6, 6, 9, 9], return 6:3, 9:2
    # output: [a value, head position, tail positon, tail-head]
    sequence = np.array(sequence)
    s1 = np.diff(sequence)
    s2 = s1 != 0
    seq_num = np.r_[True, s2, True]  # same as np.concatenate((a1, a2))
    idx = np.flatnonzero(seq_num)  # indices of nonzero elements
    s1 = np.diff(idx)
    s2 = sequence[idx[:-1]]
    seq_num = np.column_stack((s2, idx[:-1], idx[1:]-1, s1))
    return seq_num


def f_test_b(ssr_own, ssr_joint, degree_free_joint, lag):
    # Granger Causality test using ssr (F statistic)
    if lag <= 0:  # checking zero denominator
        lag = 1
    if ssr_joint == 0:  # checking zero denominator
        ssr_joint = 0.001
    fgc2 = (ssr_own - ssr_joint) / ssr_joint
    fgc1 = degree_free_joint * fgc2 / lag
    # F-test by 'ssr'
    p_value = stats.f.sf(fgc1, lag, degree_free_joint)
    return p_value


def granger_test_b(x, lag, addconst=True, no_lag=0):
    # response/target variable d_target is from the 1st column of x
    d_target, d_lag_own, d_lag_joint = make_lagged_data(x, lag, addconst)
    df_joint = len(d_target) - d_lag_joint.shape[1]
    # OLS: Fit a linear model using Ordinary Least Squares
    # bivariate Autoregressive Model with lag variables
    result_own = OLS(d_target, d_lag_own).fit()
    # regressive + Autoregressive (AR) Model with lag variables
    result_joint = OLS(d_target, d_lag_joint).fit()
    # checking target variable: most data are zeros, then no relation
    nzeros = findzeros_variable(d_target)
    if nzeros >= 0.85:
        ssr_own = 0
        ssr_joint = 9
    else:
        r2, r2_adj, ssr_own = r2_adjust(d_target, d_lag_own, result_own.params)
        r2, r2_adj, ssr_joint = r2_adjust(d_target, d_lag_joint, result_joint.params)
    p_value = f_test_b(ssr_own, ssr_joint, df_joint, lag)
    return p_value


def series_relation_test_b(da_yx, da_xy, head, tail, lag, signif_thres, lag_add=1):
    # test causual relation between two time series X & Y
    window_min = 3 * lag + 2
    if tail - head < window_min:
        raise ValueError("Min observations are needed from {0} to {1}".format(head, tail))
    # result: 1 for relation X->Y, -1 for Y->X, 0 for no relation
    zone_id = 0  # no relation
    p_value_zone = granger_test_b(da_yx, lag, True, lag_add)
    if p_value_zone < signif_thres and p_value_zone != -1:
        zone_id = 1  # causual direction X->Y
    else:
        p_value_zone = granger_test_b(da_xy, lag, True, lag_add)
        if p_value_zone < signif_thres and p_value_zone != -1:
            zone_id = -1  # Y->X
    return zone_id, p_value_zone
