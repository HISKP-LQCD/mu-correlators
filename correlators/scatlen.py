#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014-2015 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as op


def a0_intercept_generator(m, w, l):
    def a0_intercept(a0):
        c1 = -2.837297
        c2 = 6.375183

        return 2 * m - w - 4 * np.pi * a0 / (m * l**3) * (
            1 + c1 * a0 / l + c2 * a0**2 / l**2
        )

    return a0_intercept

def compute_a0(m, w, l, fig=None, x=np.linspace(-5, 5, 100)):
    a0_intercept = a0_intercept_generator(m, w, l)
    a0 = op.newton(a0_intercept, 0)

    if fig is not None:
        y = a0_intercept(x)
        fig.plot(x, y, color='blue', alpha=0.5)
        fig.plot([a0], [a0_intercept(a0)], linestyle='none', marker='.', color='black')

    return a0
