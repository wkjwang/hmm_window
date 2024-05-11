# -*- coding: utf-8 -*-
# Created by Kaijun WANG in May 2024

import numpy as np
import pandas as pd
import math


def synthetic_data(std_a=0.01, tlag=2, nsample=1000, s=450, t=650):
    # generate synthetic 2 time series，casual period [s,t]，delay tlag
    bk = np.random.randint(0, 9, 2, dtype='l')
    # low 0 is inclusive, high 10 is exclusive
    irand = np.random.randint(0, 10, 9, dtype='l') - 4
    s = s - 1 + irand[bk[0]]
    t = t + irand[bk[1]]

    num_t = np.array([i for i in range(nsample)])
    causal_label = np.zeros(nsample, dtype=int)  # 0: no causality
    ak = np.array([0.2 * pow(-1, i+1) for i in range(tlag)])
    bk = np.array([0.2] * tlag)
    # std_a = math.sqrt(var_b)

    # time series Y
    da_y = np.random.uniform(0, 0.1, nsample)
    # time series X，variation by sin，Gaussian noise, std: Standard deviation
    da_x = 1 + np.sin(0.08 * num_t) + np.random.normal(0, std_a, nsample)

    if tlag <= 0:
        tlag2 = 2
    else:
        tlag2 = tlag + tlag
    # part1: no relation [0,s-1]
    for i in range(tlag, t+tlag2):  # +lag: it continues for points around t
        y_lag = da_y[i-tlag: i]
        da_y[i] = np.dot(ak, y_lag) + np.random.normal(0, std_a)
    # smooth several starting points
    da_y[0:tlag2] = da_y[tlag2+1:tlag2+tlag2+1]

    # part2: causal part [s,t-1]
    causal_label[s:t+1] = 1
    for i in range(s, t):
        y_lag = da_y[i-tlag: i]
        x_lag = da_x[i-tlag: i]
        da_y[i] = np.dot(ak, y_lag) + np.dot(bk, x_lag) + np.random.normal(0, std_a)

    # part3: no relation [t,nsample-1]
    # [t:t+lag] continuation from part1, exclude influence of part2 points
    for i in range(t+tlag, nsample):
        y_lag = da_y[i - tlag:i]
        da_y[i] = np.dot(ak, y_lag) + np.random.normal(0, std_a)

    da_x = np.array([da_x], dtype=np.float32)
    da_y = np.array([da_y], dtype=np.float32)
    data_yx = np.hstack((da_y.T, da_x.T))
    print('-->synthetic time series：length {}，noise std {}，causal period [{},{}]；'.format(nsample, std_a, s, t))
    # first column of data_yx is the response/target variable
    return data_yx, causal_label



def dataload_taxitrips(sw=0):
    data = pd.read_csv('series_taxitrips_b.txt', sep='\t')
    data = np.array(data, dtype=np.float32)
    nsize = data.shape
    causal_label = data[:, 3]
    if sw:  # Dropoff --> Tweet
        data_yx = data[:, 0:2]
    else:  # Pickup <-- Tweet
        data_yx = data[:, [2,0]]
    for i in range(0, nsize[0]):
        data_yx[i, 0] = math.sqrt(data_yx[i, 0])
        data_yx[i, 1] = math.sqrt(data_yx[i, 1])
    timelag = 5
    window = [12, 18, 24, 48]
    step = [4, 8, 12]
    return data_yx, causal_label, timelag  # window, step



def p_sqrt(data):
    ns = data.shape
    for j in range(0, ns[1]):
        for i in range(0, ns[0]):
            sg = np.sign(data[i, j])
            data[i, j] = sg * math.sqrt(sg * data[i, j])
    return data
