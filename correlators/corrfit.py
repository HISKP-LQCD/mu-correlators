#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2015 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

'''
Fitting correlated data with least squares.
'''

import numpy as np


def correlation_matrix(sets):
    '''
    Computes the correlation matrix from a set of (multiple) time series.

    The input must have dimension 2. The first index shall label the
    measurement, the second one the time.
    '''
    N = len(sets)
