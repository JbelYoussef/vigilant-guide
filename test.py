from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 10:25:45 2016

@author: sbouazza020116
"""

import numpy as np
from numba import jit, int64, float64, boolean

@jit(boolean[:](int64[:], int64[:]), nopython=True)
def in_1d_arr(arr1, arr2):
    N = arr1.size
    M = arr2.size
    res = np.empty(N, dtype=np.bool8)
    for i in xrange(N):
        res[i] = False
        for j in xrange(M):
            if arr2[j] == arr1[i]:
                res[i] = True
                break
    return res

@jit(float64[:,:](int64[:], int64[:], int64, float64[:,:],
     float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:]), nopython=True)
def compute_quantities(calc_cal, effective_rebalancing_cal, smooth_steps,
                       weights, prices, prices_fixings, fx_coeffs,
                       fx_coeffs_fixings, nosh_adj):
    N, M = weights.shape
    quantities = np.zeros((N, M))
    index_level = np.zeros(N)
    is_effective_rebalancing_date = in_1d_arr(calc_cal, effective_rebalancing_cal)
    # Finding the index of the 1st effective rebalancing date
    for j in xrange(N):
        if is_effective_rebalancing_date[j]:
            break
    index_level[j-1] = 1000.
    quantities[j] = index_level[j-1]*weights[j]/(prices[j-1]*fx_coeffs[j-1])*nosh_adj[j]/nosh_adj[j-1]
    index_level[j] = np.sum(quantities[j]*prices[j]*fx_coeffs[j])
    for i in xrange(j+1,N):
        if is_effective_rebalancing_date[i-1]:
            quantities[i] = index_level[i-1]*weights[i]/(prices_fixings[i-1]*fx_coeffs_fixings[i-1])
        else:
            quantities[i] = quantities[i-1]
        quantities[i] *= nosh_adj[i]/nosh_adj[i-1]
        index_level[i] = np.sum(quantities[i]*prices[i]*fx_coeffs[i])
    return quantities
    # the result needs to be rounded after the call because np.round is not
    # supported in numba
