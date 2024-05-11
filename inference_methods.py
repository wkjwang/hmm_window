# -*- coding: utf-8 -*-
# Created by Kaijun WANG in May 2024

from causality_detect_functions import *
from regression_withlag import regression_lag


def hmm_states(dat_yx, lag_max, nsample, ns, step, wide=50, wide_neglect=15):
    # find hidden states corresponding to regressive relation between variables
    window_min = 3 * lag_max + 2
    r2adj = np.zeros([nsample, 4], float)
    n_stop = 0
    head = 0
    wide2 = int(wide * 0.5)
    wide3 = wide2 - 2
    while n_stop < 2:
        tail = head + wide + lag_max
        if tail-head < window_min:
            head = head + step
            raise ValueError("Min {0} observations are needed for lag {1}".format(window_min, lag_max))
            continue
        r2adj_own, r2adj_joint = regression_lag(dat_yx[head:tail, :], lag=lag_max)
        position = head + wide2 - 2  # the middle part of a sliding window
        posit_end = position + step + 1
        r2adj[position:posit_end, 0] = r2adj_own
        r2adj[position:posit_end, 1] = r2adj_joint
        if head < wide3:  # for the initial part of time series
            r2adj[head:wide3, 0] = r2adj_own
            r2adj[head:wide3, 1] = r2adj_joint
        head = head + step
        if head + wide + lag_max > nsample:
            head = nsample - wide - lag_max
            n_stop += 1

    # HMM hidden states, corresponding to different regions of time series
    hide_states = run_HMM(r2adj[:, 1])
    r2adj[:, 2] = hide_states
    # output: a state, head position, tail position, tail - head
    cseq = count_consecutive_values(hide_states[0:ns])
    idx = cseq[:, 3] > wide_neglect
    cregion = cseq[idx, 1:]
    return cregion, r2adj


def causual_bystates(dat_yx, dat_xy, lag_max, cregion, ns, labels, p_thred=0.05):
    # fiind causual relations for every region of time series, regions are from HMM hidden states
    window_min = 3 * lag_max + 2
    lab_pred = np.zeros(ns, int)
    k = -1
    head = lag_max
    for start, ends, wide in cregion:
        k = k + 1
        if (ends + 1 - head + lag_max) < window_min or wide < 30:
        # a small region is merged with its neighbor or solved by causual_regionedge()
            continue
        tail = ends + 1
        head = head - lag_max
        if tail-head < window_min:
            raise ValueError("Min {0} observations are needed for lag {1}".format(window_min, lag_max))
        # avalue: 1 for relation X->Y, -1 for Y->X, 0 for no relation
        avalue, p_val = series_relation_test_b(dat_yx[head:tail, :], dat_xy[head:tail, :], head, tail, lag_max, p_thred)
        lab_pred[head:tail] = avalue
        cregion[k, 0] = avalue
        if avalue:
            avalue = 1  # with relation
        else:
            avalue = -1  # without relation

        wide = tail - head
        if wide > 100:
            if wide > 200:
                wide = 200
            wide2 = int(wide / 2)
            step2 = int(wide2 / 2)
            wide3 = int(wide / 4)
            step3 = int(wide3 / 2)
        elif wide > 60:
            wide2 = int(wide / 2)
            step2 = int(wide2 / 2)
            wide3 = int(wide / 3)
            step3 = int(wide3 / 2)
        else:
            wide2 = 40
            step2 = 20
            wide3 = 30
            step3 = 15
        if wide2 < window_min:
            head = tail
            continue
        if wide3 < window_min:
            wide3 = window_min
        if tail-head < window_min:
            continue
        lab_pred2 = slide_window(dat_yx[head:tail, :], dat_xy[head:tail, :], head, tail, lag_max, wide2, step2)
        lab_pred3 = slide_window(dat_yx[head:tail, :], dat_xy[head:tail, :], head, tail, lag_max, wide3, step3)
        # synthesize the results of avalue, lab_pred2 and lab_pred3 by voting
        lab_pred2 = avalue + lab_pred2 + lab_pred3
        idx = lab_pred2 > 0
        lab_pred3[:] = 0  # without relation
        lab_pred3[idx] = 1  # with relation
        lab_pred[head:tail] = lab_pred3
        head = tail
    lab_pred = np.abs(lab_pred)
    acy_hmm = accuracy_rate(lab_pred, labels)
    return acy_hmm, lab_pred


def causual_regionedge(dat_yx, dat_xy, lag_max, cregion, ns, step, labels, lab_pred, p_thred=0.05):
	# Fine-tuning of test results for the provisional region centered on the segmented region edge
    window_min = 3 * lag_max + 2
    lab_pred_edge = lab_pred.copy()
    posit_indx = cregion.copy()
    p_vals = np.ones(ns, float)
    k = 0
    for avalue, ends, wide in cregion:
        # prepare the data around a sliding window
        if wide <= 40:
            wide2 = 40
        else:
            wide2 = 50
        if wide2 < window_min:
            wide2 = window_min
        # starting point
        w_shift = int(wide2 * 0.1)
        s_head = ends - wide2 - lag_max + w_shift
        if s_head < 0:
            s_head = 0
        s_tail = ends + 1 + wide2
        if s_tail > ns:
            s_tail = ns + 1
        # record the middle position of a sliding window
        w_shift = int(wide2 * 0.5) - int(step * 0.5)
        position = s_head + w_shift
        posit_indx[k, 0] = position
        posit_indx[k, 1] = position + wide2

        # causality detection around the window, using sliding window with wide2
        head = s_head
        while 1:
            tail = head + wide2 + 1
            if tail > s_tail:
                break
            position = head + w_shift
            posit_end = position + step + 1
            if head == 0:
                position = 0
            if tail-head < window_min:
                raise ValueError("Min {0} observations are needed for lag {1}".format(window_min, lag_max))
                continue
            avalue, p_val = series_relation_test_b(dat_yx[head:tail], dat_xy[head:tail], head, tail, lag_max, p_thred)
            # record predicted causal label
            lab_pred_edge[position:posit_end] = avalue
            p_vals[position:posit_end] = p_val
            position = posit_end
            head = head + step
        k = k + 1

    lab_pred_edge = np.abs(lab_pred_edge)
    acy_slide_edge = accuracy_rate(lab_pred_edge, labels)
    return acy_slide_edge, p_vals


def slide_window(dat_yx, dat_xy, ihead, itail, lag_max, awide, step, p_thred=0.05):
    # a sliding window with size of awide moves forward in step
    window_min = 3 * lag_max + 2
    ndata = itail - ihead
    if ndata < window_min:
        raise ValueError("Min observations are needed from {0} to {1}".format(ihead, itail))
    lab_pred = np.zeros(ndata, int)
    p_vals = np.ones(ndata, float)
    s_tail = ndata
    head = 0
    position = int(awide * 0.5) - int(step * 0.5)
    astop = 1
    while astop:
        tail = head + awide + 1
        posit_end = position + step + 1
        if tail > s_tail:
            tail = ndata
            posit_end = tail
            head = tail - awide - 1
            astop = 0
        if head == 0:
            position = 0
        if tail-head < window_min:
            raise ValueError("Min observations are needed from {0} to {1}".format(head, tail))
            continue
        avalue, p_val = series_relation_test_b(dat_yx[head:tail], dat_xy[head:tail], head, tail, lag_max, p_thred)
        # record predicted causal label
        lab_pred[position:posit_end] = avalue
        p_vals[position:posit_end] = p_val
        position = posit_end
        head = head + step

    lab_pred = np.abs(lab_pred)
    idx = lab_pred < 0.9  # it is same as using < 1
    lab_pred[idx] = -1  # without relation
    return lab_pred

