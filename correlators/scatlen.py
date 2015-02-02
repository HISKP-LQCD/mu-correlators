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

def compute_a0(m, w, l):
    a0_intercept = a0_intercept_generator(m, w, l)

    try:
        a0 = op.newton(a0_intercept, 0)
    except RuntimeError as e:
        x = np.linspace(-50, 50, 100)
        y = a0_intercept(x)

        fig = pl.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, marker='+')
        ax.grid(True)
        fig.show()

        raw_input()

        raise

    return a0
