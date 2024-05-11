# -*- coding: utf-8 -*-
# Created by Kaijun WANG in May 2024
# The codes are for the paper: Kaijun WANG, and etc. Detection Windows from Hidden Markov Model
# for Discovering Varying Causal Relations Between Time Series

# note: the code prefers small lag, e.g., lag_max <= 5, although it is workable for lag_max > 5.
# The code has no special design for good running under a big lag. If you need lag_max > 5, 
# it is recommended to adjust the data to meet lag_max <= 5.

from data_causaltimeseries import *
from inference_methods import *
# from causality_detect_functions import *


a_datset = 1
window_size = [20, 40, 80]
step = 5
ftest_thred = 0.05
if a_datset == 1:
    n_repeat = 2
else:
    n_repeat = 1
acy_matrix = np.zeros([2, n_repeat], float)

for k in range(n_repeat):
    if a_datset == 1:
        ns = 500
        ns_ext = 0
        lag_max = 2
        dat_yx, labels = synthetic_data(0.10, lag_max, ns, s=150, t=350)  #450,650
'''
    elif a_datset == 2:
        dat_yx, labels, lag_max = dataload_taxitrips(1)  # Dropoff-->Tweet
        ns = dat_yx.shape[0]
        ns_ext = 0
    elif a_datset == 3:
        dat_yx, labels, lag_max = dataload_taxitrips(0)  # Pickup<--Tweet
        ns = dat_yx.shape[0]
        ns_ext = 0
'''
    # prepare the dataset by variable position exchange
    nsample = ns + ns_ext
    dat_xy = dat_yx[0:ns, :].copy()
    dat_xy[:, 0] = dat_yx[0:ns, 1]
    dat_xy[:, 1] = dat_yx[0:ns, 0]

    # HMM produces consecutive-state regions, corresponding to segmented regions of time series
    cregion, r2adj = hmm_states(dat_yx, lag_max, nsample, ns, step)
    # Causality testing in segmented regions
    acy_hmm, lab_hmm = causual_bystates(dat_yx, dat_xy, lag_max, cregion, ns, labels)
    # Fine-tuning of test results
    acy_edge, pv_edge = causual_regionedge(dat_yx, dat_xy, lag_max, cregion, ns, step, labels, lab_hmm)
    acy_matrix[0, k] = acy_hmm
    acy_matrix[1, k] = acy_edge
    print(acy_matrix[:, k])
    np.savetxt('output_accuracy.txt', acy_matrix, fmt="%.4f", delimiter="\t")

print('-------- Code running is over --------')
